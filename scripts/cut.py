r"""Cut Claude's picks into finished vertical clips.

Two steps, because Claude looks at a frame of each clip in between:

  python cut.py download --picks ~/Clipper/picks.json
      downloads ONLY the minutes each pick needs, saves three frames and a contact
      sheet per clip, and creates a run folder with layout.json for Claude to fill in

  python cut.py download --run "~/Clipper/runs/<run>" --only 03
      re-downloads one pick after you change its start/end in that run's picks.json

  python cut.py render --run "~/Clipper/runs/<run>"
      builds each 1080x1920 clip: face cam on top, gameplay underneath,
      a hook for the first 3 seconds, word-by-word captions, Shorts-level audio,
      then checks every clip (length, black face cam, file written)
"""
import argparse
import csv
import datetime as dt
import os
import pathlib
import re
import sys
import shutil
import tempfile

import yt_dlp
from yt_dlp.utils import download_range_func

from common import FFMPEG, HOME_DIR, WORK, explain_download_error, hms, load, media_duration, parse_time, publish_video, read_wav, run, save, slug, ydl_opts

PAD = 1.5          # extra seconds downloaded on each side, trimmed exactly at render
OUT_W, OUT_H = 1080, 1920
TOP_H = 768        # face cam band (40%)
HOOK_SECONDS = 3.0
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    str(HOME_DIR / "fonts" / "Anton-Regular.ttf"),  # downloaded by setup.sh, for Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


# ---------------------------------------------------------------- download

def fetch_pick(run_dir, p, key):
    """Download one pick's minutes into raw/<key>.mp4 and save its check frames. Returns True on success."""
    meta = load(WORK / "streams" / str(p.get("source")) / "meta.json")
    if not meta:
        print(f"!! {key}: unknown source {p.get('source')!r}. Use the stream id from meta.json, "
              "and run analyze.py on that stream first. Skipping.")
        return False
    try:
        start, end = parse_time(p["start"]), parse_time(p["end"])
    except (KeyError, ValueError):
        print(f"!! {key}: start/end missing or not a time (use HH:MM:SS). Skipping.")
        return False
    if not 8 <= end - start <= 90:
        print(f"!! {key}: length {end - start:.0f}s is outside 8 to 90 seconds. Skipping.")
        return False
    if meta.get("duration") and end > meta["duration"] + 1:
        print(f"!! {key}: ends at {hms(end)} but the stream is only {hms(meta['duration'])} long. Skipping.")
        return False
    p.update({"key": key, "start_s": start, "end_s": end, "url": meta["url"],
              "streamer": p.get("streamer") or meta.get("channel")})
    raw_dir = run_dir / "raw"
    for old in raw_dir.glob(f"{key}.*"):  # a re-download replaces the old minutes
        old.unlink()
    raw = raw_dir / f"{key}.mp4"
    print(f"==> {key}: {hms(start)} to {hms(end)} from {meta.get('channel')}")
    opts = ydl_opts(
        format="bv*[height<=1080][vcodec^=avc1]+ba/bv*[height<=1080]+ba/b[height<=1080]/b",
        outtmpl=str(raw_dir / f"{key}.%(ext)s"),
        merge_output_format="mp4",
        download_ranges=download_range_func(None, [(max(0, start - PAD), end + PAD)]),
        force_keyframes_at_cuts=True,
    )
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([meta["url"]])
    except yt_dlp.utils.DownloadError as e:
        print(f"!! {key}: {explain_download_error(e)} Skipping.")
        return False
    if not raw.exists():
        got = [f for f in raw_dir.glob(f"{key}.*") if f.suffix not in (".part", ".ytdl")]
        if not got:
            print(f"!! {key}: download failed, skipping")
            return False
        got[0].rename(raw)
    p["lead_in"] = min(PAD, start)
    # three frames (start, middle, end) so Claude can see if the stream layout changes inside the clip
    clip_len = end - start
    for tag, t in (("a", 1.0), ("b", clip_len / 2), ("c", clip_len - 1.0)):
        run([FFMPEG, "-v", "error", "-y", "-ss", f"{p['lead_in'] + max(0.0, t):.2f}", "-i", str(raw),
             "-frames:v", "1", "-vf", "scale=960:-2", str(run_dir / "frames" / f"{key}_{tag}.jpg")], check=False)
    # contact sheet: one small frame every 2 seconds, left to right, top to bottom,
    # so Claude can see exactly when a layout switch happens and trim around it
    tiles = max(1, int(clip_len // 2))
    rows = (tiles + 5) // 6
    run([FFMPEG, "-v", "error", "-y", "-ss", f"{p['lead_in']:.2f}", "-t", f"{clip_len:.2f}", "-i", str(raw),
         "-vf", f"fps=1/2,scale=320:-2,tile=6x{rows}:padding=4:color=white", "-frames:v", "1",
         str(run_dir / "frames" / f"{key}_sheet.jpg")], check=False)
    return True


def cmd_download(a):
    if a.run:
        # re-download some picks of an existing run (after changing their start/end in <run>/picks.json)
        run_dir = pathlib.Path(os.path.expanduser(a.run))
        doc = load(run_dir / "picks.json")
        if not doc:
            raise SystemExit(f"No picks.json in {run_dir}")
        picks = doc["picks"]
        only = {k.zfill(2) for k in (a.only or [p["key"] for p in picks])}
        layout = load(run_dir / "layout.json", {}) or {}
        for p in picks:
            if p["key"] in only:
                if fetch_pick(run_dir, p, p["key"]):
                    layout.setdefault(p["key"], {"facecam": None, "focus_x": 0.5, "note": "fill in after looking at the frames"})
                    print(f"   frames/{p['key']}_a.jpg  _b.jpg  _c.jpg  _sheet.jpg")
        save(run_dir / "picks.json", {"picks": picks})
        save(run_dir / "layout.json", layout)
        print(f"\nRun folder: {run_dir}\nLook at the new frames, update layout.json if the layout changed, "
              f"then render with --only {' '.join(sorted(only))}")
        return

    picks_doc = load(a.picks)
    if not picks_doc:
        raise SystemExit(f"No picks file at {a.picks}")
    picks = picks_doc["picks"] if isinstance(picks_doc, dict) else picks_doc
    run_dir = WORK / "runs" / dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)

    layout = {}
    clean = []
    for n, p in enumerate(picks, 1):
        key = f"{n:02d}"
        if fetch_pick(run_dir, p, key):
            layout[key] = {"facecam": None, "focus_x": 0.5, "note": "fill in after looking at the frames"}
            clean.append(p)

    save(run_dir / "picks.json", {"picks": clean})
    save(run_dir / "layout.json", layout)
    if not clean:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise SystemExit("\nNo clips downloaded (see the !! lines above). Fix picks.json and run this again.")
    print(f"\nRun folder: {run_dir}")
    print("NEXT: look at all three frames of every clip in frames/ (start, middle, end), then fill in layout.json:")
    print('  "facecam": [x, y, w, h] as fractions of the frame (0 to 1), or null if there is no face cam')
    print('  "focus_x": where the action is horizontally in the gameplay, 0 = left, 0.5 = center, 1 = right')
    print("  If the stream layout changes between the frames, change that pick's start/end in the run's picks.json")
    print(f"  and run: cut.py download --run \"{run_dir}\" --only <key>")
    for p in clean:
        print(f"  frames/{p['key']}_a.jpg  {p['key']}_b.jpg  {p['key']}_c.jpg  ({p['key']}_sheet.jpg = one frame every 2s)")


# ---------------------------------------------------------------- render helpers

def video_size(path):
    r = run([FFMPEG, "-hide_banner", "-i", str(path)], check=False)
    m = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", r.stderr or "")
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


def has_audio(path):
    r = run([FFMPEG, "-hide_banner", "-i", str(path)], check=False)
    return "Audio:" in (r.stderr or "")


def has_encoder(name):
    r = run([FFMPEG, "-hide_banner", "-encoders"], check=False, capture=True)
    return name in (r.stdout or "")


def font(size):
    from PIL import ImageFont
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def words_for_clip(raw, lead_in, dur, model):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        run([FFMPEG, "-v", "error", "-y", "-ss", str(lead_in), "-t", str(dur), "-i", str(raw),
             "-ac", "1", "-ar", "16000", wav])
        segs, _ = model.transcribe(read_wav(wav), word_timestamps=True, vad_filter=True, beam_size=1)
        words = []
        for s in segs:
            for w in s.words or []:
                text = re.sub(r"[^\w'!?$%.,-]", "", w.word.strip())
                if text:
                    words.append((max(0.0, w.start), min(dur, w.end), text))
    return words


def caption_chunks(words, max_words=3, max_chars=16):
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        text = " ".join(x[2] for x in cur)
        if (len(cur) >= max_words or len(text) >= max_chars or w[2][-1:] in ".!?," or not nxt
                or nxt[0] - w[1] > 0.6):
            chunks.append([cur[0][0], cur[-1][1], text.upper().strip(".,")])
            cur = []
    for i in range(len(chunks) - 1):  # hold each chunk until the next one starts (no flicker)
        if chunks[i + 1][0] - chunks[i][1] < 0.6:
            chunks[i][1] = chunks[i + 1][0]
    return chunks


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_overlay(path, hook, caption, cap_y):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if hook:
        f = font(66)
        lines = wrap(d, hook.upper(), f, OUT_W - 160)
        lh = 80
        box_h = lh * len(lines) + 40
        y0 = 120
        d.rounded_rectangle([60, y0, OUT_W - 60, y0 + box_h], radius=24, fill=(255, 214, 10, 255))
        for i, ln in enumerate(lines):
            w = d.textlength(ln, font=f)
            d.text(((OUT_W - w) / 2, y0 + 20 + i * lh), ln, font=f, fill=(10, 10, 10, 255))
    if caption:
        f = font(84)
        lines = wrap(d, caption, f, OUT_W - 120)
        for i, ln in enumerate(lines):
            w = d.textlength(ln, font=f)
            d.text(((OUT_W - w) / 2, cap_y + i * 100), ln, font=f, fill=(255, 255, 255, 255),
                   stroke_width=8, stroke_fill=(0, 0, 0, 255))
    img.save(path)


def build_overlay_track(tmp, hook, chunks, dur, cap_y):
    times = {0.0, dur, min(HOOK_SECONDS, dur)}
    for c in chunks:
        times.update([max(0.0, min(dur, c[0])), max(0.0, min(dur, c[1]))])
    times = sorted(times)
    cache, lines = {}, []
    for a, b in zip(times, times[1:]):
        if b - a < 0.02:
            continue
        mid = (a + b) / 2
        h = hook if mid < HOOK_SECONDS else ""
        cap = next((c[2] for c in chunks if c[0] <= mid < c[1]), "")
        k = (h, cap)
        if k not in cache:
            cache[k] = os.path.join(tmp, f"ov{len(cache):04d}.png")
            render_overlay(cache[k], h, cap, cap_y)
        lines += [f"file '{cache[k]}'", f"duration {b - a:.3f}"]
    lines.append(lines[-2])  # concat demuxer needs the last file repeated
    lst = os.path.join(tmp, "overlay.txt")
    with open(lst, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lst


def layout_filter(src_w, src_h, lay):
    fc = lay.get("facecam")
    fx = min(1.0, max(0.0, float(lay.get("focus_x", 0.5) or 0.5)))
    if fc:
        x, y, w, h = [min(1.0, max(0.0, float(v))) for v in fc]
        x, y = min(x, 0.95), min(y, 0.95)
        cx, cy = int(x * src_w), int(y * src_h)
        cw, ch = max(16, int(w * src_w)), max(16, int(h * src_h))
        cw, ch = min(cw, src_w - cx), min(ch, src_h - cy)
        gw = min(src_w, int(src_h * OUT_W / (OUT_H - TOP_H)) // 2 * 2)
        gx = int(min(max(0, fx * src_w - gw / 2), src_w - gw))
        return (f"[0:v]split=2[a][b];"
                f"[a]crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{TOP_H}:force_original_aspect_ratio=increase,"
                f"crop={OUT_W}:{TOP_H}[top];"
                f"[b]crop={gw}:{src_h}:{gx}:0,scale={OUT_W}:{OUT_H - TOP_H}[bot];"
                f"[top][bot]vstack=inputs=2,setsar=1[base]"), TOP_H + 70
    gw = min(src_w, int(src_h * OUT_W / OUT_H) // 2 * 2)
    gx = int(min(max(0, fx * src_w - gw / 2), src_w - gw))
    return f"[0:v]crop={gw}:{src_h}:{gx}:0,scale={OUT_W}:{OUT_H},setsar=1[base]", 1250


def frame_brightness(path, t, box):
    from PIL import Image, ImageStat
    with tempfile.TemporaryDirectory() as tmp:
        jpg = os.path.join(tmp, "f.jpg")
        run([FFMPEG, "-v", "error", "-y", "-ss", str(t), "-i", str(path), "-frames:v", "1", jpg], check=False)
        if not os.path.exists(jpg):
            return 0.0
        return sum(ImageStat.Stat(Image.open(jpg).convert("L").crop(box)).mean)


def render_one(p, raw, out, dur, lay, model, venc):
    words = words_for_clip(raw, p["lead_in"], dur, model)
    chunks = caption_chunks(words)
    src_w, src_h = video_size(raw)
    vf, cap_y = layout_filter(src_w, src_h, lay)
    with tempfile.TemporaryDirectory() as tmp:
        lst = build_overlay_track(tmp, p.get("hook", ""), chunks, dur, cap_y)
        audio_in = "[0:a]" if has_audio(raw) else "[2:a]"  # silent track if the source has no audio
        fc = (f"{vf};[1:v]format=rgba[ov];[base][ov]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[v];"
              f"{audio_in}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
        silent = [] if audio_in == "[0:a]" else ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        run([FFMPEG, "-v", "error", "-y", "-ss", str(p["lead_in"]), "-t", f"{dur:.3f}", "-i", str(raw),
             "-f", "concat", "-safe", "0", "-i", lst, *silent,
             "-filter_complex", fc, "-map", "[v]", "-map", "[aout]", *venc,
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-t", f"{dur:.3f}", out])

    # quality checks, on a frame after the hook is gone and on areas the captions don't cover
    problems = []
    got = media_duration(out)
    if abs(got - dur) > 1.0:
        problems.append(f"length {got:.1f}s, expected {dur:.1f}s")
    t_check = min(max(HOOK_SECONDS + 1.0, dur / 2), dur - 0.5)
    if lay.get("facecam"):
        if frame_brightness(out, t_check, (0, 0, OUT_W, TOP_H)) < 12:
            problems.append("face cam area looks black, check the facecam box in layout.json")
        game_box = (0, TOP_H + 420, OUT_W, OUT_H)
    else:
        game_box = (0, 360, OUT_W, 1200)
    if frame_brightness(out, t_check, game_box) < 6:
        problems.append("gameplay area looks black")
    if not words:
        problems.append("no speech found, so no captions (fine for a pure reaction scream)")
    return problems, got, words


# ---------------------------------------------------------------- render

def cmd_render(a):
    from faster_whisper import WhisperModel
    run_dir = os.path.expanduser(a.run)
    doc = load(os.path.join(run_dir, "picks.json"))
    if not doc:
        sys.exit(f"No picks.json in {run_dir}. Use the run folder that cut.py download printed.")
    picks = doc["picks"]
    only = {k.zfill(2) for k in a.only} if a.only else None
    layout = load(os.path.join(run_dir, "layout.json"), {})
    out_dir = os.path.join(run_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    if has_encoder("libx264"):
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    elif has_encoder("h264_videotoolbox"):
        venc = ["-c:v", "h264_videotoolbox", "-b:v", "8M", "-pix_fmt", "yuv420p"]
    else:
        sys.exit("This ffmpeg has no H.264 encoder. Run setup.sh again.")

    previous = {c["key"]: c for c in (load(os.path.join(run_dir, "clips.json"), {}) or {}).get("clips", [])}
    results, rendered = [], []
    for p in picks:
        key = p["key"]
        if only and key not in only:
            if key in previous:
                results.append(previous[key])
            continue
        raw = os.path.join(run_dir, "raw", f"{key}.mp4")
        dur = p["end_s"] - p["start_s"]
        lay = layout.get(key, {})
        name = f"{key}_{slug(p.get('hook') or p.get('title') or key)}.mp4"
        out = os.path.join(out_dir, name)
        print(f"==> {key}: rendering {dur:.0f}s clip ({'face cam on top' if lay.get('facecam') else 'full-frame crop'})")
        if not os.path.exists(raw):
            print(f"!! {key}: no downloaded video at {raw}. Run cut.py download again for it. Skipping.")
            continue
        try:
            problems, got, words = render_one(p, raw, out, dur, lay, model, venc)
        except Exception as e:  # one broken clip shouldn't stop the others
            print(f"!! {key}: render failed: {str(e).strip().splitlines()[-1][:300]}")
            continue
        for old_clip in os.listdir(out_dir):  # a re-render with a new hook gets a new file name
            if old_clip.startswith(f"{key}_") and old_clip != name:
                os.remove(os.path.join(out_dir, old_clip))
        status = "ok" if not problems else "; ".join(problems)
        print(f"   {status}")
        video = publish_video(out, run_dir, key, slug(p.get('hook') or p.get('title') or key))
        print(f"   saved: {video}")
        p.update({"file": f"clips/{name}", "qc": status, "duration": round(got, 1), "video": video})
        results.append(p)
        rendered.append(p)

    save(os.path.join(run_dir, "clips.json"), {"clips": results})
    # log.csv feeds the weekly "look at what's working" report. Re-rendering a clip replaces its row.
    log = WORK / "log.csv"
    header = ["date", "run", "file", "source_url", "start", "streamer", "moment_type", "score", "hook", "title"]
    run_name = os.path.basename(os.path.normpath(run_dir))
    rows = []
    if log.exists():
        with open(log, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    redone = {(run_name, p["key"]) for p in rendered}
    rows = [r for r in rows if (r.get("run"), os.path.basename(r.get("file", ""))[:2]) not in redone]
    for p in rendered:
        rows.append(dict(zip(header, [dt.date.today().isoformat(), run_name, p["file"], p["url"], hms(p["start_s"]),
                                      p.get("streamer", ""), p.get("moment_type", ""), p.get("score", ""),
                                      p.get("hook", ""), p.get("title", "")])))
    with open(log, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDONE: {len(rendered)} clips rendered, {len(results)} in {out_dir}")
    print(f"Next: python gallery.py --run \"{run_dir}\"")
    if a.delete_raw:
        shutil.rmtree(os.path.join(run_dir, "raw"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--picks", default=str(WORK / "picks.json"))
    d.add_argument("--run", help="re-download picks of this existing run (after editing its picks.json)")
    d.add_argument("--only", nargs="*", help="with --run: just these keys, e.g. --only 03")
    r = sub.add_parser("render")
    r.add_argument("--run", required=True)
    r.add_argument("--model", default="small", help="whisper model for captions (small is accurate; base is faster)")
    r.add_argument("--only", nargs="*", help="re-render just these keys, e.g. --only 03 07")
    r.add_argument("--delete-raw", action="store_true", help="delete the downloaded source minutes afterwards")
    a = ap.parse_args()
    cmd_download(a) if a.cmd == "download" else cmd_render(a)


if __name__ == "__main__":
    main()
