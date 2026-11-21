"""Listen to a stream and find the moments worth clipping.

Steps: download ONLY the audio, read it once to get per-second signals (the streamer's voice,
laughs and screams, game sounds), read the live-chat replay for real reactions (laughing, shock,
fails, hype; greetings, "lag?", polls and commands are ignored), shift chat back by its measured
delay, build one hype score per second, then transcribe a window around each top moment with
Whisper on your own computer.

With --visual it also downloads a 144p copy and watches the screen (motion bursts, and screens going
dark or gray, which can be death and fail screens). On YouTube it reads the top comments for timestamps viewers bookmarked.

Usage:
  python analyze.py <stream url> [--top 40] [--model small] [--full] [--visual] [--no-chat] [--no-sounds]
                    [--no-comments] [--game gta] [--chat-lag 25]

Output (in ~/Clipper/streams/<id>/):
  for_claude.md    every candidate moment: when it peaks, what chat said, a suggested cut, and the
                   transcript with an intensity bar per line (Claude reads this)
  candidates.json  the same, as data
  report.html      the timeline with every signal and every candidate, to look through yourself
  hype.png         the hype curve picture
  meta.json        stream title, channel, duration, url
  features.npz     cached per-second audio signals, so a re-run skips the slow part
"""
import argparse
import bisect
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter

import numpy as np
import yt_dlp

import hype
from report import write_report
from profiles import detect_profile, keyword_score
from sound_events import stream_features
from common import WORK, FFMPEG, explain_download_error, hms, load, load_whisper, read_wav, run, save, stream_dir, ydl_opts

FEATURES_VERSION = 3  # 3: adds music


def get_info(url):
    with yt_dlp.YoutubeDL(ydl_opts()) as ydl:
        return ydl.extract_info(url, download=False)


def download_audio(url, d):
    for f in d.glob("audio.*"):
        if f.suffix not in (".part", ".ytdl"):
            return f
    opts = ydl_opts(format="bestaudio/worstaudio/worst", outtmpl=str(d / "audio.%(ext)s"))
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return next(f for f in d.glob("audio.*") if f.suffix not in (".part", ".ytdl"))


def audio_features(audio, d, with_model):
    """Per-second audio signals, cached in features.npz (an 11-hour stream takes about 3 minutes)."""
    cache = d / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if int(z["version"]) == FEATURES_VERSION and (bool(z["model"]) or not with_model):
            feat = {k: z[k] for k in z.files if k not in ("version", "model", "labels")}
            if "labels" in z.files:
                feat["labels"] = [str(x) for x in z["labels"]]
            print("   (using the saved audio signals)")
            return feat
    feat = stream_features(audio, with_model)
    extra = {"labels": np.array(feat["labels"])} if "labels" in feat else {}
    np.savez_compressed(cache, version=FEATURES_VERSION, model="speech" in feat,
                        **{k: v for k, v in feat.items() if k != "labels"}, **extra)
    return feat


def download_chat(url, d):
    path = d / "chat.live_chat.json"
    if path.exists():
        return path
    opts = ydl_opts(skip_download=True, writesubtitles=True, subtitleslangs=["live_chat"], outtmpl=str(d / "chat"))
    for attempt in (1, 2):  # long chat replays sometimes cut off halfway; start over once
        for part in d.glob("chat*.part"):
            part.unlink()
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:  # chat replay is a bonus, never a blocker
            print(f"   (chat replay attempt {attempt} failed: {str(e)[:80]})")
        if path.exists():
            return path
    return None


def fetch_comments(url, limit=500):
    """Top comments, for the timestamps viewers leave on the best moments. Never a blocker."""
    opts = ydl_opts(skip_download=True, getcomments=True,
                    extractor_args={"youtube": {"max_comments": [str(limit), str(limit), "0", "0"], "comment_sort": ["top"]}})
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False).get("comments") or []
    except Exception as e:
        print(f"   (couldn't read the comments: {str(e)[:80]})")
        return []


def visual_features_cached(url, d, n):
    import visual
    cache = d / "visual.npz"
    if cache.exists():
        z = np.load(cache)
        print("   (using the saved screen signals)")
        return {k: z[k] for k in z.files}
    print("   downloading a 144p copy of the stream (about the size of the audio)")
    video = visual.download_low_video(url, d)
    vf = visual.visual_features(video, n)
    if vf is not None:
        print("   finding the face cam (a few frames at 720p)")
        box = visual.find_stream_facecam(url, d, n)
        if box:
            fm = visual.facecam_motion(video, box, n)
            if fm is not None:
                vf["facecam_motion"] = fm
        np.savez_compressed(cache, **vf)
    return vf


def heatmap_per_second(info, n):
    hm = info.get("heatmap") or []
    if not hm:
        return None
    v = np.zeros(n)
    for seg in hm:
        a, b = int(seg["start_time"]), int(math.ceil(seg["end_time"]))
        v[max(0, a):min(n, b)] = np.maximum(v[max(0, a):min(n, b)], seg.get("value", 0))
    return v


def detect_language(model, audio, windows, tmp, tries=3):
    """The stream's language, by majority over a few windows. Whisper guessing per window flips to
    Arabic or Korean on a loud or short window and writes nonsense, so it's decided once."""
    votes = Counter()
    for i, w in enumerate(windows[:tries]):
        wav = os.path.join(tmp, f"lang{i}.wav")
        run([FFMPEG, "-v", "error", "-y", "-ss", str(w["start"]), "-t", str(min(30.0, w["end"] - w["start"])),
             "-i", str(audio), "-ac", "1", "-ar", "16000", wav])
        lang, prob, _ = model.detect_language(read_wav(wav))
        votes[lang] += prob
    return votes.most_common(1)[0][0] if votes else None


def transcribe(audio, windows, model_name, d=None, language=None):
    """Whisper on each window, all in one language (detected once, saved in meta.json). Spans already
    transcribed with the same model (saved in transcripts.json) are reused, so re-running the
    analysis doesn't transcribe them again."""
    cache_path = d / "transcripts.json" if d else None
    cache = (load(cache_path, []) or []) if cache_path else []
    meta_path = d / "meta.json" if d else None
    language = language or ((load(meta_path, {}) or {}).get("language") if meta_path else None)
    model, out, fresh = None, [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, w in enumerate(windows):
            hit = next((c for c in cache if c["model"] == model_name and c["start"] <= w["start"] + 0.5
                        and c["end"] >= w["end"] - 0.5), None)
            if hit:
                out.append([s for s in hit["segs"] if w["start"] - 1 <= s["start"] < w["end"]])
                continue
            if model is None:
                model = load_whisper(model_name)
                if not language:
                    language = detect_language(model, audio, windows, tmp)
                    print(f"   language: {language}")
                    if meta_path and meta_path.exists():
                        save(meta_path, {**load(meta_path, {}), "language": language})
            wav = os.path.join(tmp, f"w{i}.wav")
            run([FFMPEG, "-v", "error", "-y", "-ss", str(w["start"]), "-t", str(w["end"] - w["start"]),
                 "-i", str(audio), "-ac", "1", "-ar", "16000", wav])
            segs, _ = model.transcribe(read_wav(wav), language=language, vad_filter=True, word_timestamps=True,
                                       beam_size=1)
            segs = [{"start": round(w["start"] + s.start, 2), "end": round(w["start"] + s.end, 2), "text": s.text.strip()}
                    for s in segs]
            out.append(segs)
            cache.append({"model": model_name, "start": w["start"], "end": w["end"], "segs": segs})
            fresh += 1
            if cache_path and fresh % 5 == 0:
                save(cache_path, cache)
            print(f"   transcribed window {i + 1}/{len(windows)} ({hms(w['start'])})", flush=True)
    if cache_path and fresh:
        save(cache_path, cache)
    if fresh < len(windows):
        print(f"   ({len(windows) - fresh} of {len(windows)} windows were already transcribed)")
    return out


def draw_hype(score, moments, path, title):
    from PIL import Image, ImageDraw
    W, H, pad = 1600, 420, 40
    n = len(score)
    cols = W - 2 * pad
    per = max(1, int(math.ceil(n / cols)))
    bars = np.array([score[i:i + per].max() for i in range(0, n, per)])
    img = Image.new("RGB", (W, H), (14, 14, 18))
    dr = ImageDraw.Draw(img)
    mx = float(bars.max()) or 1.0
    for i, s in enumerate(bars):
        x = pad + int(i / max(1, len(bars) - 1) * cols)
        h = int(s / mx * (H - 2 * pad - 30))
        dr.line([(x, H - pad), (x, H - pad - h)], fill=(255, 92, 56) if s > mx * 0.5 else (90, 90, 110))
    for t in moments:
        x = pad + int(t / max(1, n - 1) * cols)
        dr.ellipse([x - 5, pad + 18, x + 5, pad + 28], fill=(255, 214, 10))
    dr.text((pad, 10), f"Hype curve: {title[:90]}", fill=(230, 230, 230))
    img.save(path)


def collect_signals(d, feat, use_chat=True, use_comments=True, use_visual=False, chat_lag=None, quiet=False):
    """Every per-second signal for a stream, from what's already saved in its folder (features.npz,
    the chat replay, comments.json, heatmap.json, visual.npz). Used by analyze.py and benchmark.py.
    Returns (signals, extras) with extras = msgs, ts, lag, marks, vlabels."""
    say = (lambda *x: None) if quiet else print
    n = len(feat["loud"])
    signals = {"voice": hype.scale(hype.voice_arousal(feat))}
    say("   + streamer's voice")
    if "reaction" in feat:
        signals["laugh_scream"] = hype.scale(hype.smooth(feat["reaction"], 3), floor=0.3)
        signals["game_action"] = hype.scale(hype.smooth(hype.onset(feat["action"]), 3), floor=0.3)
        say("   + laughs, screams and game sounds")
    ex = {"msgs": [], "ts": [], "lag": 0, "marks": [], "vlabels": None, "already": []}
    if use_visual and (d / "visual.npz").exists():
        import visual
        z = np.load(d / "visual.npz")
        vsig, ex["vlabels"] = visual.visual_signal({k: z[k] for k in z.files})
        signals["visual"] = hype.scale(vsig)
        say("   + what happens on screen")
        if "facecam_motion" in z.files:
            fm = np.clip(hype.robust_z(hype.smooth(z["facecam_motion"], 2)), 0, None)
            signals["facecam_motion"] = hype.scale(hype.smooth(fm, 3))
            say("   + the streamer moving in the face cam")
    if use_comments and (d / "comments.json").exists():
        csig, ex["marks"] = hype.comment_signal(load(d / "comments.json", []), n)
        if len(ex["marks"]) >= 3:
            signals["viewer_bookmarks"] = hype.scale(csig, floor=1.0)
            say(f"   + {len(ex['marks'])} viewer bookmarks")
        else:
            say(f"   ({len(ex['marks'])} timestamps in the comments, not enough to use)")
    clips = load(d / "community_clips.json", None)
    if clips:
        signals["community_clips"] = hype.scale(hype.clip_signal(clips, n), floor=1.0)
        say(f"   + {len(clips)} Twitch community clips")
    ex["already"] = [x for x in (load(d / "already_clipped.json", []) or []) if x.get("confidence", 99) >= 15]
    if ex["already"]:
        signals["already_clipped"] = hype.scale(hype.clip_signal(ex["already"], n), floor=1.0)
        say(f"   + {len(ex['already'])} Shorts other clippers already posted from this stream")
    heat = heatmap_per_second({"heatmap": load(d / "heatmap.json", [])}, n)
    if heat is not None and heat.max() > 0:
        signals["most_replayed"] = heat / heat.max()
        say("   + YouTube most-replayed graph")
    cp = d / "chat.live_chat.json"
    if use_chat and cp.exists():
        counts, total, ex["msgs"] = hype.read_chat(cp, n)
        ex["ts"] = [m[0] for m in ex["msgs"]]
        burst = hype.chat_burst(counts)
        ex["lag"] = chat_lag if chat_lag is not None else hype.estimate_lag(burst, signals["voice"])
        signals["chat_reactions"] = hype.scale(hype.shift_back(burst, ex["lag"]))
        react = int(sum(c.sum() for c in counts.values()))
        say(f"   + chat reactions ({react:,} of {int(total.sum()):,} messages are reactions, the rest is greetings, "
            f"'lag?', polls and talk; chat runs {ex['lag']}s behind the stream)")
    return signals, ex


def chat_evidence(msgs, ts, start, end):
    """What chat was saying about a moment: counts by kind and a few messages worth reading
    (reactions, plus longer plain messages, which often say what happened)."""
    part = msgs[bisect.bisect_left(ts, start):bisect.bisect_right(ts, end)]
    kinds = Counter(m[2] for m in part if m[2])
    seen, react, plain = set(), [], []
    for _, text, kind in part:
        key = re.sub(r"(.)\1{2,}", r"\1\1", text.lower())
        if key in seen or not text:
            continue
        seen.add(key)
        if kind and len(react) < 5:
            react.append(text)
        elif not kind and len(text.split()) >= 3 and len(plain) < 5:
            plain.append(text)
    return dict(kinds), react + plain


def suggest_cut(score, moment, segs, n):
    """A starting cut for Claude, on sentence boundaries: start at the line that sets the moment up
    (one that follows a pause or a finished sentence, about 8 seconds before the peak), end on the
    first finished sentence after the reaction dies down. Never mid-word."""
    react_end = float(hype.moment_end(score, moment))
    segs = sorted(segs, key=lambda s: s["start"])
    start, end = moment - 8.0, react_end + 1.0
    best = None
    for i, s in enumerate(segs):
        if not moment - 20 <= s["start"] <= moment - 2:
            continue
        gap = s["start"] - segs[i - 1]["end"] if i else 9.0
        prev_done = i == 0 or segs[i - 1]["text"].rstrip()[-1:] in ".?!"
        quality = (1.0 if gap >= 0.6 else 0.0) + (0.5 if prev_done else 0.0) - abs(s["start"] - (moment - 8)) / 8
        if best is None or quality > best[0]:
            best = (quality, s["start"])
    if best:
        start = best[1] - 0.3
    for i, s in enumerate(segs):
        if s["end"] < react_end - 1 or s["end"] > react_end + 12:
            continue
        gap = segs[i + 1]["start"] - s["end"] if i + 1 < len(segs) else 9.0
        if s["text"].rstrip()[-1:] in ".?!" or gap >= 0.8:
            end = s["end"] + 0.4
            break
    end = min(max(end, start + 12), start + 50, n)
    return max(0.0, start), end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--top", type=int, default=40, help="how many candidate moments to send to Claude")
    ap.add_argument("--model", default="small", help="whisper model: base (fast) or small (better)")
    ap.add_argument("--full", action="store_true", help="transcribe the whole stream, not just the hot windows")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--no-sounds", action="store_true", help="skip the sound classifier (speech, laughs, gunfire...)")
    ap.add_argument("--game", help="game word list to use (gta, valorant...); picked from the title when not given")
    ap.add_argument("--chat-lag", type=int, help="seconds chat lags behind the stream (measured when not given)")
    ap.add_argument("--visual", action="store_true", help="also watch a 144p copy: motion bursts, screens going dark or gray")
    ap.add_argument("--no-comments", action="store_true", help="don't read viewer timestamps from the comments")
    ap.add_argument("--deep", action="store_true", help="vocal separation and speaker diarization on the candidates "
                    "(needs setup.sh --deep; about 20 s per candidate on a CPU)")
    a = ap.parse_args()
    t0 = time.time()

    print("==> Reading stream info")
    try:
        info = get_info(a.url)
    except yt_dlp.utils.DownloadError as e:
        sys.exit(explain_download_error(e))
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        sys.exit("This stream is still live (or its recording isn't processed yet). "
                 "Clip it once it ends and the full recording is up. On YouTube that's usually "
                 "right after the stream ends; 'post_live' can take a few hours to finish.")
    vid = info["id"]
    d = stream_dir(vid)
    duration = float(info.get("duration") or 0)
    meta = {
        "id": vid, "url": info.get("webpage_url") or a.url, "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"), "duration": duration,
        "extractor": info.get("extractor_key"), "was_live": info.get("was_live"),
    }
    save(d / "meta.json", meta)
    print(f"   {meta['channel']} | {meta['title']} | {hms(duration)}")

    print("==> Downloading audio only")
    try:
        audio = download_audio(a.url, d)
    except yt_dlp.utils.DownloadError as e:
        sys.exit(explain_download_error(e))
    print(f"   {audio.name}: {audio.stat().st_size / 1e6:.0f} MB")

    print("==> Listening to every second (voice, laughs and screams, game sounds)")
    feat = audio_features(audio, d, with_model=not a.no_sounds)
    n = len(feat["loud"])
    if n < 5:
        sys.exit(f"The audio file looks empty or broken. Delete {d} and run this again.")
    if not duration:
        duration = float(n)
        meta["duration"] = duration
        save(d / "meta.json", meta)
    short = duration < 1800

    profile = detect_profile(meta["title"], a.game)
    if profile:
        print(f"   + {profile} word list for the transcripts")
    if not a.no_comments and meta["extractor"] == "Youtube" and not (d / "comments.json").exists():
        print("==> Reading the top comments for timestamps viewers bookmarked")
        save(d / "comments.json", fetch_comments(meta["url"]))
    if a.visual and not (d / "visual.npz").exists():
        print("==> Watching the screen (motion bursts, screens going dark or gray)")
        try:
            visual_features_cached(meta["url"], d, n)
        except yt_dlp.utils.DownloadError as e:
            print(f"   (couldn't download the low-res video, skipping the screen: {explain_download_error(e)})")
    if not a.no_chat and meta["extractor"] == "Youtube" and info.get("was_live"):
        if "live_chat" not in (info.get("subtitles") or {}):
            print("   (this stream's chat replay is turned off, so it uses the audio only)")
        elif not (d / "chat.live_chat.json").exists():
            print("==> Reading the live chat replay (long streams can take a few minutes)")
            download_chat(a.url, d)
    elif not a.no_chat and meta["extractor"] != "Youtube":
        print(f"   ({meta['extractor']} has no chat replay to read, so it uses the audio only)")
    save(d / "heatmap.json", info.get("heatmap") or [])
    if "twitch" in (meta["extractor"] or "").lower() and not (d / "community_clips.json").exists():
        import twitch
        try:
            clips = twitch.community_clips(meta["url"])
            if clips is None:
                print("   (set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET to also use this VOD's community clips)")
            else:
                save(d / "community_clips.json", clips)
        except Exception as e:  # a bonus signal, never a blocker
            print(f"   (couldn't read the Twitch community clips: {str(e)[:100]})")

    signals, ex = collect_signals(d, feat, use_chat=not a.no_chat, use_comments=not a.no_comments,
                                  use_visual=a.visual, chat_lag=a.chat_lag)
    msgs, ts, lag, marks, vlabels = ex["msgs"], ex["ts"], ex["lag"], ex["marks"], ex["vlabels"]
    allsegs = None
    if a.full:  # transcribe everything first, so what's said can count toward the score
        chunks = [{"start": float(s), "end": float(min(duration, s + 600))} for s in range(0, int(math.ceil(duration)), 600)]
        print(f"==> Transcribing the whole stream ({len(chunks)} x 10 min) with Whisper ({a.model})")
        allsegs = [s for part in transcribe(audio, chunks, a.model, d) for s in part]
        signals["words"] = hype.scale(hype.words_signal(allsegs, n, profile), floor=1.0)
        print("   + reaction and game words, every second")

    weights = load(WORK / "weights.json", None)
    if weights:
        print(f"   (using your channel's signal weights from {WORK / 'weights.json'})")
    score = hype.combine(signals, weights)
    moments = hype.pick_peaks(score, a.top, 30 if short else 90)
    if not moments:
        sys.exit("No hype spikes found (the stream may be silent or very short). Try --full to read all of it.")
    # the yell itself: the loudest voice second near the combined peak
    v = signals["voice"]
    moments = [int(max(range(max(0, m - 6), min(n, m + 7)), key=lambda t: v[t])) if v[m] > 0 else m for m in moments]
    before, after = (25, 15) if short else (45, 25)

    if allsegs is not None:
        segs_per = [[s for s in allsegs if m - before <= s["start"] <= m + after] for m in moments]
    else:
        windows = [{"start": max(0.0, m - before), "end": min(float(n), m + after)} for m in moments]
        print(f"==> Transcribing {len(windows)} moments with Whisper ({a.model})")
        segs_per = transcribe(audio, windows, a.model, d)

    top_score = max(score[m] for m in moments) or 1.0
    cands = []
    for m, ss in zip(moments, segs_per):
        kinds, said = chat_evidence(msgs, ts, m + lag - 5, m + lag + 20) if msgs else ({}, [])
        words = keyword_score(" ".join(s["text"] for s in ss), profile)
        why = []
        if signals.get("chat_reactions") is not None and signals["chat_reactions"][m] > 0.4:
            why.append("chat " + ", ".join(f"{cnt} {kind}" for kind, cnt in sorted(kinds.items(), key=lambda kv: -kv[1])))
        if v[m] > 0.5:
            why.append("streamer yelling" if v[m] > 1 else "streamer worked up")
        if "laugh_scream" in signals and signals["laugh_scream"][m] > 0.5:
            why.append("laugh/scream heard")
        if feat.get("labels") and feat["labels"][m]:
            why.append(f"sound: {feat['labels'][m]}")
        if "most_replayed" in signals and signals["most_replayed"][m] > 0.5:
            why.append("most replayed")
        if "community_clips" in signals and signals["community_clips"][m] > 0.3:
            why.append("viewers clipped it on Twitch")
        done = [x for x in ex["already"] if x["offset"] - 20 <= m <= x["offset"] + x.get("duration", 30) + 5]
        music = feat.get("music")
        if music is not None and np.mean(music[max(0, m - 20):min(n, m + 20)] > 0.5) > 0.5:
            why.append("music playing most of the time (copyright risk: mute it or cut around it)")
        if vlabels and any(vlabels[t] for t in range(max(0, m - 3), min(n, m + 4))):
            why.append("screen: " + next(vlabels[t] for t in range(max(0, m - 3), min(n, m + 4)) if vlabels[t]))
        near = sorted([mk for mk in marks if abs(mk[0] - m) <= 30], key=lambda mk: -mk[1])[:3]
        if near:
            why.append(f"{len(near)} viewer bookmark{'s' if len(near) > 1 else ''} in the comments")
        arc, arc_a, arc_b = hype.arc_length(score, m)
        kind = hype.moment_kind(kinds, float(v[m]), float(signals.get("laugh_scream", np.zeros(n))[m]),
                                (vlabels[m] if vlabels else ""))
        start, end = suggest_cut(score, m, ss, n)
        cands.append({
            "moment": m, "start": round(max(0.0, m - before), 1), "end": round(min(float(n), m + after), 1),
            "rank": float(score[m]) * (1 + 0.06 * words),
            "signals": {k: round(float(s[m]), 2) for k, s in signals.items()},
            "why": why, "chat_kinds": kinds, "chat_said": said, "keywords": words, "kind": kind,
            "arc": int(arc), "arc_span": [int(arc_a), int(arc_b)], "bookmarks": near,
            "already_clipped": [{"title": x["title"], "views": x["views"], "url": x["url"]} for x in done],
            "suggested": [round(start, 1), round(end, 1)],
            "segments": [{"t": round(s["start"], 1), "level": hype.level_bar(score[min(n - 1, int(s["start"]))], top_score),
                          "text": s["text"]} for s in ss],
        })
    if a.deep:
        import deep
        if not deep.available():
            print("   (--deep needs the deep extras: run bash scripts/setup.sh --deep. Skipping it.)")
        else:
            print(f"==> Deep listening: separating the voices and telling speakers apart ({len(cands)} moments)")
            deep.analyze_windows(audio, cands)
            for c in cands:
                c["rank"] *= 1 + 0.4 * min(2.0, c.get("vocal", 0))
                if c.get("vocal_laugh_shout", 0) > 0.4:
                    c["why"].append(f"laughing/yelling in the separated voice ({c['vocal_laugh_shout']:.2f})")
                if c.get("speakers", 0) >= 2 and c.get("turns_per_min", 0) >= 8:
                    c["why"].append(f"fast back-and-forth between {c['speakers']} speakers ({c['turns_per_min']:.0f} turns/min)")
                    if c["kind"] == "talk":
                        c["kind"] = "argument or banter"
    for c in cands:
        at = max([i for i, s in enumerate(c["segments"]) if s["t"] <= c["moment"]], default=0)
        for i, s in enumerate(c["segments"]):
            s["moment"] = i == at
    cands.sort(key=lambda c: -c["rank"])
    best = cands[0]["rank"] or 1.0
    for i, c in enumerate(cands, 1):
        c["cand"], c["strength"] = i, int(round(100 * c["rank"] / best))
    save(d / "candidates.json", {"meta": meta, "signals_used": list(signals), "chat_lag": lag, "profile": profile,
                                 "candidates": cands})
    draw_hype(score, moments, d / "hype.png", meta["title"] or vid)
    write_report(d / "report.html", meta, score, signals, cands, lag)

    lines = [f"# {meta['title']}", f"Channel: {meta['channel']} | Length: {hms(duration)} | URL: {meta['url']}",
             f"Signals used: {', '.join(signals)}" + (f" | chat runs {lag}s behind the stream" if msgs else "")
             + (f" | word list: {profile}" if profile else ""), "",
             "Each candidate is one moment, strongest first. MOMENT is the second the stream peaks (chat is "
             "already shifted back by its delay, so this is when it happened on screen, not when chat typed). "
             "Strength is 0 to 100 against this stream's best moment. 'looks like' is a first guess at the kind of moment. 'chat said' is what viewers typed right "
             "after it: it usually tells you what happened. 'suggested' is a starting cut from the setup to where "
             "the reaction dies down; tighten it. The bar on each line is how intense that second was "
             "(▮▮▮▮▮ = as intense as the best moment).", ""]
    for c in cands:
        lines.append(f"## Candidate {c['cand']} | MOMENT {hms(c['moment'])} | strength {c['strength']} | "
                     f"looks like: {c['kind']} | words {c['keywords']:.0f}")
        if c["why"]:
            lines.append("why: " + "; ".join(c["why"]))
        if c["arc"] >= 45:
            lines.append(f"arc: the hype stays up for {c['arc']}s ({hms(c['arc_span'][0])} to {hms(c['arc_span'][1])}); "
                         "a longer 45 to 90 second clip may tell the whole story")
        for mk in c["bookmarks"]:
            lines.append(f'viewer bookmark ({mk[1]} likes): "{mk[2]}"')
        for x in c["already_clipped"]:
            lines.append(f'already clipped by others ({x["views"]:,} views): "{x["title"]}" {x["url"]}')
        if c["chat_said"]:
            lines.append("chat said: " + " / ".join(f'"{s}"' for s in c["chat_said"]))
        lines.append(f"suggested: {hms(c['suggested'][0])} to {hms(c['suggested'][1])}")
        for s in c["segments"]:
            who = f"{s['speaker']}: " if s.get("speaker") else ""
            lines.append(f"[{hms(s['t'])} {s['level']}] {who}{s['text']}" + ("  <-- MOMENT" if s["moment"] else ""))
        if not c["segments"]:
            lines.append("(no speech: could be a pure gameplay, laugh or scream moment)")
        lines.append("")
    if a.full:
        lines.append("\n# Full transcript in 12-minute blocks\n")
        block = -1
        for s in allsegs:
            if int(s["start"] // 720) != block:
                block = int(s["start"] // 720)
                lines.append(f"\n## Block {block + 1} ({hms(block * 720)} to {hms((block + 1) * 720)})")
            lines.append(f"[{hms(s['start'])} {hype.level_bar(score[min(n - 1, int(s['start']))], top_score)}] {s['text']}")
    (d / "for_claude.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")
    print(f"Stream folder: {d}")
    print(f"Claude reads:  {d / 'for_claude.md'}")
    print(f"Report:        {d / 'report.html'}")
    print(f"Hype curve:    {d / 'hype.png'}")


if __name__ == "__main__":
    main()
