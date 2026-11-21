"""Find moments while a YouTube stream is still live, minutes after they happen.

  python live.py <live url> [--every 300] [--model base] [--hours 6]

Records the stream's audio (from its lowest-quality feed) and its live chat from now on, and every
--every seconds scores what's new with the same signals as analyze.py (the streamer's voice, laughs
and screams, chat reactions shifted back by chat's delay). Each strong new moment is transcribed
right away and appended to ~/Clipper/streams/<id>/live_moments.md, and to candidates.json so
cut.py can clip it: times are in stream time, and cut.py downloads them from the live DVR.
Stop with Ctrl+C, or it stops when the stream ends.
"""
import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import yt_dlp

import hype
from analyze import chat_evidence, transcribe
from common import FFMPEG, HOME_DIR, hms, load, media_duration, save, stream_dir, ydl_opts
from sound_events import stream_features


def live_info(url):
    with yt_dlp.YoutubeDL(ydl_opts()) as ydl:
        return ydl.extract_info(url, download=False)


def lowest_feed(info):
    """The smallest HLS feed of the live stream that has audio: an audio-only feed if there is one,
    otherwise the lowest-quality feed with sound."""
    fs = [f for f in info.get("formats", []) if f.get("protocol", "").startswith("m3u8") and f.get("url")]
    audio_only = [f for f in fs if f.get("vcodec") == "none"]  # YouTube lists these with an unknown acodec
    with_sound = [f for f in fs if f.get("acodec") not in (None, "none")]
    pool = audio_only or with_sound
    if not pool:
        return None
    return min(pool, key=lambda f: (f.get("height") or 0, f.get("tbr") or 1e9))["url"]


def start_recording(feed, path):
    """ffmpeg copies the audio into an MPEG-TS file, which can be read while it grows."""
    return subprocess.Popen([FFMPEG, "-v", "error", "-y", "-i", feed, "-vn", "-c:a", "copy", "-f", "mpegts", str(path)],
                            stdin=subprocess.DEVNULL)


def start_chat(url, d):
    ytdlp = HOME_DIR / "venv" / "bin" / "yt-dlp"
    cmd = [str(ytdlp), "--skip-download", "--write-subs", "--sub-langs", "live_chat", "-o", str(d / "live"), url]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def slice_features(ts, start, end, tmp):
    wav = os.path.join(tmp, "slice.wav")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(ts),
                    "-ac", "1", "-ar", "16000", wav], check=False)
    return stream_features(wav) if os.path.exists(wav) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--every", type=int, default=300, help="seconds between checks")
    ap.add_argument("--model", default="base", help="whisper model for each new moment (base keeps up best)")
    ap.add_argument("--hours", type=float, default=6, help="stop after this long")
    ap.add_argument("--min-strength", type=float, default=1.2, help="how strong a moment must be to report")
    a = ap.parse_args()

    info = live_info(a.url)
    if not info.get("is_live"):
        sys.exit("This stream isn't live. For a finished stream use analyze.py.")
    feed = lowest_feed(info)
    if not feed:
        sys.exit("Couldn't find a feed to record for this live stream.")
    d = stream_dir(info["id"])
    t_rec = time.time()
    stream_start = info.get("release_timestamp") or t_rec
    offset = max(0.0, t_rec - stream_start)  # how far into the stream the recording starts
    meta = {"id": info["id"], "url": info.get("webpage_url") or a.url, "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"), "duration": 0, "extractor": info.get("extractor_key"),
            "was_live": True, "live": True, "record_offset": offset, "record_start_epoch": t_rec}
    save(d / "meta.json", meta)
    ts = d / "live_audio.ts"
    rec = start_recording(feed, ts)
    chat = start_chat(meta["url"], d) if info.get("extractor_key") == "Youtube" else None
    print(f"==> Recording {meta['channel']} | {meta['title']}")
    print(f"   recording starts {hms(offset)} into the stream; checking every {a.every}s. Ctrl+C to stop.")

    feats, done_until, reported = {}, 0.0, []
    md = d / "live_moments.md"
    if not md.exists():
        md.write_text(f"# Live moments: {meta['title']}\n{meta['url']}\n\n", encoding="utf-8")
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    deadline = t_rec + a.hours * 3600
    with tempfile.TemporaryDirectory() as tmp:
        while not stop["now"] and time.time() < deadline:
            for _ in range(a.every):
                if stop["now"] or rec.poll() is not None:
                    break
                time.sleep(1)
            have = media_duration(ts) if ts.exists() else 0.0
            if have - done_until >= 30:
                f = slice_features(ts, done_until, have, tmp)
                if f:
                    for k, v in f.items():
                        if k != "labels":
                            feats[k] = np.concatenate([feats.get(k, np.zeros(0)), v])
                    done_until += len(f["loud"])
            n = len(feats.get("loud", []))
            if n >= 120:
                signals = {"voice": hype.scale(hype.voice_arousal(feats))}
                if "reaction" in feats:
                    signals["laugh_scream"] = hype.scale(hype.smooth(feats["reaction"], 3), floor=0.3)
                msgs, tss, lag = [], [], 0
                cp = d / "live.live_chat.json"
                if not cp.exists():  # yt-dlp keeps writing to .part until the stream ends
                    cp = d / "live.live_chat.json.part"
                if cp.exists():
                    counts, _, msgs = hype.read_chat(cp, n)
                    tss = [m[0] for m in msgs]
                    burst = hype.chat_burst(counts)
                    lag = hype.estimate_lag(burst, signals["voice"], peaks=15)
                    signals["chat_reactions"] = hype.scale(hype.shift_back(burst, lag))
                score = hype.combine(signals)
                limit = n - max(30, lag + 10)  # leave time for chat to react to the newest seconds
                fresh = [m for m in hype.pick_peaks(score[:limit], 20, 90)
                         if score[m] >= a.min_strength and all(abs(m - r) >= 90 for r in reported)]
                for m in sorted(fresh):
                    reported.append(m)
                    segs = transcribe(ts, [{"start": max(0.0, m - 45), "end": min(float(n), m + 25)}], a.model, d)[0]
                    kinds, said = chat_evidence(msgs, tss, m + lag - 5, m + lag + 20) if msgs else ({}, [])
                    t_stream = m + offset
                    lines = [f"## {hms(t_stream)} (stream time) | strength {score[m]:.1f} | found {time.strftime('%H:%M')}"]
                    if said:
                        lines.append("chat said: " + " / ".join(f'"{x}"' for x in said))
                    lines += [f"[{hms(s['start'] + offset)}] {s['text']}" for s in segs] + [""]
                    with open(md, "a", encoding="utf-8") as fh:
                        fh.write("\n".join(lines) + "\n")
                    doc = load(d / "candidates.json", {"meta": meta, "candidates": []})
                    doc["candidates"].append({"moment": int(t_stream), "strength": int(round(100 * min(1, score[m] / 3))),
                                              "chat_kinds": kinds, "chat_said": said, "signals": {}})
                    save(d / "candidates.json", doc)
                    print(f"   MOMENT {hms(t_stream)}  strength {score[m]:.1f}  {' / '.join(said[:3])}", flush=True)
                meta["duration"] = offset + n
                save(d / "meta.json", meta)
            if rec.poll() is not None:
                print("   the stream ended (or the feed stopped)")
                break
    rec.terminate()
    if chat:
        chat.terminate()
    print(f"\nMoments: {md}\nClip them with cut.py (source \"{meta['id']}\"); times are stream time.")


if __name__ == "__main__":
    main()
