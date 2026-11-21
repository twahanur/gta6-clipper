"""Listen to a stream and find the moments worth clipping.

Steps: download ONLY the audio, measure loudness every second, pull YouTube's
"most replayed" graph and the live-chat replay (when they exist), build one hype
curve, then transcribe the hottest windows with Whisper on your own computer.

Every second is also run through a sound classifier (sound_events.py) that hears gunfire,
explosions, crashes, sirens, screams and laughter, and the transcripts get a bonus for reaction
words (and game words when the title names a game, see profiles.py).

Usage:
  python analyze.py <stream url> [--top 40] [--model small] [--full] [--no-chat] [--game gta]

Output (in ~/Clipper/streams/<id>/):
  for_claude.md    every candidate window with its transcript and hype score (Claude reads this)
  candidates.json  the same, as data
  hype.png         the hype curve picture
  meta.json        stream title, channel, duration, url
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import yt_dlp

from profiles import detect_profile, keyword_score
from sound_events import event_signal, top_labels
from common import FFMPEG, explain_download_error, hms, read_wav, run, save, stream_dir, ydl_opts

BIN = 10  # seconds per hype bin
CHAT_WORDS = ("lol", "lmao", "lmfao", "clip", "omg", "wtf", "???", "kekw", "w ", "no way", "bro", "😂", "💀")


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


def loudness_per_second(audio):
    """dB loudness for every second of the stream, streamed so a 6-hour file never sits in memory."""
    sr = 8000
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         stdout=subprocess.PIPE)
    out = []
    while True:
        buf = p.stdout.read(sr * 2)
        if not buf:
            break
        x = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        rms = math.sqrt(float(np.mean(x * x))) if len(x) else 0.0
        out.append(20 * math.log10(rms / 32768 + 1e-9))
    p.wait()
    return np.array(out)


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


def chat_bins(path, nbins):
    counts = np.zeros(nbins)
    hits = np.zeros(nbins)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
                r = o["replayChatItemAction"]
                t = int(r["videoOffsetTimeMsec"]) / 1000
                item = r["actions"][0]["addChatItemAction"]["item"]
                msg = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
                if not msg:
                    continue
                text = "".join(x.get("text", "") for x in msg.get("message", {}).get("runs", [])).lower()
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            b = int(t // BIN)
            if 0 <= b < nbins:
                counts[b] += 1
                if any(w in text + " " for w in CHAT_WORDS):
                    hits[b] += 1
    return counts + 2 * hits


def heatmap_bins(info, nbins):
    hm = info.get("heatmap") or []
    if not hm:
        return None
    v = np.zeros(nbins)
    for seg in hm:
        a, b = int(seg["start_time"] // BIN), int(math.ceil(seg["end_time"] / BIN))
        v[max(0, a):min(nbins, b)] = np.maximum(v[max(0, a):min(nbins, b)], seg.get("value", 0))
    return v


def robust_z(x, window=30):
    """How unusual each bin is compared to the stream's own normal around that time."""
    z = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        lo, hi = max(0, i - window), min(len(x), i + window)
        ref = x[lo:hi]
        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-6
        z[i] = (x[i] - med) / mad
    return np.clip(z, 0, 6)


def pick_windows(score, duration, top, before=70, after=50, spacing=120):
    order = np.argsort(-score)
    taken = []
    for b in order:
        t = b * BIN + BIN / 2
        if score[b] <= 0:
            break
        if all(abs(t - u) >= spacing for u in taken):
            taken.append(t)
        if len(taken) >= top:
            break
    wins = []
    for t in sorted(taken):
        wins.append({"peak": t, "start": max(0.0, t - before), "end": min(duration, t + after)})
    return wins


def transcribe(audio, windows, model_name):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, w in enumerate(windows):
            wav = os.path.join(tmp, f"w{i}.wav")
            run([FFMPEG, "-v", "error", "-y", "-ss", str(w["start"]), "-t", str(w["end"] - w["start"]),
                 "-i", str(audio), "-ac", "1", "-ar", "16000", wav])
            segs, _ = model.transcribe(read_wav(wav), vad_filter=True, word_timestamps=True, beam_size=1)
            w_segs = []
            for s in segs:
                w_segs.append({
                    "start": w["start"] + s.start,
                    "end": w["start"] + s.end,
                    "text": s.text.strip(),
                })
            out.append(w_segs)
            print(f"   transcribed window {i + 1}/{len(windows)} ({hms(w['start'])})", flush=True)
    return out


def draw_hype(score, windows, path, title):
    from PIL import Image, ImageDraw
    W, H, pad = 1600, 420, 40
    img = Image.new("RGB", (W, H), (14, 14, 18))
    dr = ImageDraw.Draw(img)
    n = len(score)
    mx = float(score.max()) or 1.0
    for i, s in enumerate(score):
        x = pad + int(i / max(1, n - 1) * (W - 2 * pad))
        h = int(s / mx * (H - 2 * pad - 30))
        dr.line([(x, H - pad), (x, H - pad - h)], fill=(255, 92, 56) if s > mx * 0.5 else (90, 90, 110))
    for w in windows:
        x = pad + int((w["peak"] / BIN) / max(1, n - 1) * (W - 2 * pad))
        dr.ellipse([x - 5, pad + 18, x + 5, pad + 28], fill=(255, 214, 10))
    dr.text((pad, 10), f"Hype curve: {title[:90]}", fill=(230, 230, 230))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--top", type=int, default=40, help="how many candidate moments to send to Claude")
    ap.add_argument("--model", default="small", help="whisper model: base (fast) or small (better)")
    ap.add_argument("--full", action="store_true", help="transcribe the whole stream, not just the hot windows")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--no-sounds", action="store_true", help="skip the sound classifier (gunfire, screams...)")
    ap.add_argument("--game", help="game word list to use (gta); picked from the title when not given")
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

    print("==> Measuring loudness every second")
    loud = loudness_per_second(audio)
    if len(loud) < 5:
        sys.exit(f"The audio file looks empty or broken. Delete {d} and run this again.")
    if not duration:
        duration = float(len(loud))
        meta["duration"] = duration
        save(d / "meta.json", meta)
    nb = int(math.ceil(len(loud) / BIN))
    loud_bins = np.array([loud[i * BIN:(i + 1) * BIN].max() for i in range(nb)])
    signals = {"loudness": robust_z(loud_bins)}
    groups = None
    if not a.no_sounds:
        print("==> Listening for action and reaction sounds (gunfire, explosions, crashes, screams, laughter)")
        ev = event_signal(audio, nb, BIN)
        if ev is not None:
            combined, groups = ev
            top = float(np.percentile(combined, 99)) or 1.0
            signals["sound_events"] = np.clip(combined / top * 4, 0, 6)
            print("   + sound events")
    profile = detect_profile(meta["title"], a.game)
    if profile:
        print(f"   + {profile} word list for the transcripts")

    heat = heatmap_bins(info, nb)
    if heat is not None and heat.max() > 0:
        signals["most_replayed"] = heat / heat.max() * 4
        print("   + YouTube most-replayed graph")

    if not a.no_chat and meta["extractor"] == "Youtube" and info.get("was_live"):
        if "live_chat" not in (info.get("subtitles") or {}):
            print("   (this stream's chat replay is turned off, so it uses loudness and most-replayed only)")
        else:
            print("==> Reading the live chat replay (long streams can take a few minutes)")
            cp = download_chat(a.url, d)
            if cp:
                signals["chat_speed"] = robust_z(chat_bins(cp, nb))
                print("   + chat speed")
    elif not a.no_chat and meta["extractor"] != "Youtube":
        print(f"   ({meta['extractor']} has no chat replay to read, so it uses loudness only)")

    weights = {"loudness": 0.30, "sound_events": 0.30, "chat_speed": 0.25, "most_replayed": 0.15}
    tot = sum(weights[k] for k in signals)
    score = sum(signals[k] * weights[k] / tot for k in signals)
    pct = (np.argsort(np.argsort(score)) / max(1, nb - 1) * 100).round()

    if a.full:
        # 10-minute chunks, so a 6-hour stream never has to sit in memory at once
        chunks = [{"peak": 0, "start": float(s), "end": float(min(duration, s + 600))}
                  for s in range(0, int(math.ceil(duration)), 600)]
        print(f"==> Transcribing the whole stream ({len(chunks)} x 10 min) with Whisper ({a.model})")
        windows = [{"peak": 0, "start": 0.0, "end": duration}]
        segs = [[s for part in transcribe(audio, chunks, a.model) for s in part]]
    else:
        # long streams: 2-minute windows spaced 2 minutes apart; short videos get tighter windows
        # so a 10-minute video still gives Claude plenty of moments to choose from
        if duration >= 1800:
            windows = pick_windows(score, duration, a.top)
        else:
            windows = pick_windows(score, duration, a.top, before=35, after=25, spacing=45)
        if not windows:
            sys.exit("No hype spikes found (the stream may be silent or very short). Try --full to read all of it.")
        print(f"==> Transcribing {len(windows)} hot windows with Whisper ({a.model})")
        segs = transcribe(audio, windows, a.model)

    cands = []
    for i, (w, ss) in enumerate(zip(windows, segs), 1):
        b = min(nb - 1, int(w["peak"] // BIN))
        cands.append({
            "cand": i, "start": round(w["start"], 1), "end": round(w["end"], 1), "peak": round(w["peak"], 1),
            "hype": int(pct[b]),
            "signals": {k: round(float(v[b]), 2) for k, v in signals.items()},
            "keywords": keyword_score(" ".join(s["text"] for s in ss), profile),
            "sounds": top_labels(groups, b),
            "segments": [{"t": round(s["start"], 1), "hype": int(pct[min(nb - 1, int(s["start"] // BIN))]),
                          "text": s["text"]} for s in ss],
        })
    if not a.full:  # strongest first: hype plus a bonus for reaction/game words in what was said
        cands.sort(key=lambda c: -(c["hype"] + 8 * c["keywords"]))
    save(d / "candidates.json", {"meta": meta, "signals_used": list(signals), "profile": profile, "candidates": cands})
    draw_hype(score, windows, d / "hype.png", meta["title"] or vid)

    lines = [f"# {meta['title']}", f"Channel: {meta['channel']} | Length: {hms(duration)} | URL: {meta['url']}",
             f"Signals used: {', '.join(signals)}" + (f" | word list: {profile}" if profile else ""), ""]
    if a.full:
        # one long window: split into 12-minute blocks so it reads in pieces
        allsegs = cands[0]["segments"]
        lines.append("Full transcript in 12-minute blocks. Hype is 0 to 100 (100 = loudest/most active part of this stream).\n")
        block = -1
        for s in allsegs:
            if int(s["t"] // 720) != block:
                block = int(s["t"] // 720)
                lines.append(f"\n## Block {block + 1} ({hms(block * 720)} to {hms((block + 1) * 720)})")
            lines.append(f"[{hms(s['t'])} | hype {s['hype']}] {s['text']}")
    else:
        lines.append("Each candidate is a window around a hype spike, strongest first. Hype is 0 to 100 "
                     "(100 = the most intense moment of this stream). 'sounds' is what the sound classifier "
                     "heard at the peak (action = gunfire/explosions/crashes/sirens, reaction = screams/shouts/"
                     "laughter, talk = plain talking, high talk with no action is usually a cutscene). "
                     "'words' counts reaction and game words said in the window.\n")
        for c in cands:
            lines.append(f"## Candidate {c['cand']} | peak {hms(c['peak'])} | hype {c['hype']} | words {c['keywords']:.0f} | {c['signals']}")
            if c["sounds"]:
                lines.append(f"sounds: {c['sounds']}")
            lines.append(f"Window {hms(c['start'])} to {hms(c['end'])}")
            for s in c["segments"]:
                lines.append(f"[{hms(s['t'])} | hype {s['hype']}] {s['text']}")
            if not c["segments"]:
                lines.append("(no speech: could be a pure gameplay, laugh or scream moment)")
            lines.append("")
    (d / "for_claude.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")
    print(f"Stream folder: {d}")
    print(f"Claude reads:  {d / 'for_claude.md'}")
    print(f"Hype curve:    {d / 'hype.png'}")


if __name__ == "__main__":
    main()
