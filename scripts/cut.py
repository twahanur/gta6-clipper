r"""Cut Claude's picks into finished vertical clips.

Two steps, because Claude looks at a frame of each clip in between:

  python cut.py download --picks ~/Clipper/picks.json
      downloads ONLY the minutes each pick needs, saves three frames and a contact
      sheet per clip, and creates a run folder with layout.json for Claude to fill in

  python cut.py download --run "~/Clipper/runs/<run>" --only 03
      re-downloads one pick after you change its start/end in that run's picks.json

  python cut.py render --run "~/Clipper/runs/<run>"
      builds each 1080x1920 clip: face cam on top, gameplay underneath,
      a hook for the first 3 seconds, word-by-word karaoke captions, Shorts-level audio,
      then checks every clip (length, black face cam, file written)
"""
import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import re
import sys
import shutil
import tempfile

import yt_dlp
from yt_dlp.utils import download_range_func

from common import (FFMPEG, WORK, clean_word, explain_download_error, hms, load, load_whisper, media_duration,
                    parse_time, publish_video, read_wav, run, save, slug, ydl_opts)
from fonts import font_for

PAD = 1.5          # extra seconds downloaded on each side, trimmed exactly at render
OUT_W, OUT_H = 1080, 1920
TOP_H = 768        # face cam band (40%)
HOOK_SECONDS = 3.0
YELLOW = (255, 214, 10, 255)
WHITE = (255, 255, 255, 255)
BLACK = (10, 10, 10, 255)


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
    p.pop("status", None)
    raw_dir = run_dir / "raw"
    for old in raw_dir.glob(f"{key}.*"):  # a re-download replaces the old minutes
        old.unlink()
    raw = raw_dir / f"{key}.mp4"
    print(f"==> {key}: {hms(start)} to {hms(end)} from {meta.get('channel')}")
    # YouTube sometimes answers 403 for its regular (DASH) files while its HLS feeds work: try both
    formats = ["bv*[height<=1080][vcodec^=avc1]+ba/bv*[height<=1080]+ba/b[height<=1080]/b",
               "b[height<=1080][protocol^=m3u8]/bv*[height<=1080][protocol^=m3u8]+ba"]
    for i, fmt in enumerate(formats):
        opts = ydl_opts(
            format=fmt,
            outtmpl=str(raw_dir / f"{key}.%(ext)s"),
            merge_output_format="mp4",
            download_ranges=download_range_func(None, [(max(0, start - PAD), end + PAD)]),
            force_keyframes_at_cuts=True,
            live_from_start=bool(meta.get("live")),  # a moment found by live.py: fetch it from the live DVR
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([meta["url"]])
            break
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            for part in raw_dir.glob(f"{key}.*"):
                part.unlink()
            if i + 1 < len(formats) and ("403" in msg or "exited with code" in msg):
                print(f"   {key}: the regular download was refused, trying YouTube's other feed")
                continue
            print(f"!! {key}: {explain_download_error(e)} Skipping.")
            p["status"] = "download failed"
            return False
    if not raw.exists():
        got = [f for f in raw_dir.glob(f"{key}.*") if f.suffix not in (".part", ".ytdl")]
        if not got:
            print(f"!! {key}: download failed, skipping")
            return False
        got[0].rename(raw)
    p["lead_in"] = min(PAD, start)
    p.pop("status", None)
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
                    if p["key"] not in layout:
                        layout[p["key"]] = auto_layout(run_dir, p["key"], p)
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
    clean, failed = [], []
    for n, p in enumerate(picks, 1):
        key = f"{n:02d}"
        p["key"] = key
        if fetch_pick(run_dir, p, key):
            layout[key] = auto_layout(run_dir, key, p)
            print(f"   layout: {layout[key]['note']}")
            clean.append(p)
        else:
            p["status"] = p.get("status") or "download failed"
            failed.append(p)

    # failed picks stay in the run's picks.json, so they can be retried with --run --only <key>
    save(run_dir / "picks.json", {"picks": sorted(clean + failed, key=lambda p: p["key"])})
    save(run_dir / "layout.json", layout)
    if failed:
        print(f"\n{len(failed)} pick(s) didn't download: {' '.join(p['key'] for p in failed)}. "
              f"Retry with: cut.py download --run \"{run_dir}\" --only {' '.join(p['key'] for p in failed)}")
    if not clean:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise SystemExit("\nNo clips downloaded (see the !! lines above). Fix picks.json and run this again.")
    print(f"\nRun folder: {run_dir}")
    print("NEXT: look at all three frames of every clip in frames/ (start, middle, end), then check layout.json")
    print("(it's pre-filled by the face cam detector; fix any box that's off):")
    print('  "facecam": [x, y, w, h] as fractions of the frame (0 to 1), or null if there is no face cam')
    print('  "focus_x": where the action is horizontally in the gameplay, 0 = left, 0.5 = center, 1 = right')
    print("  If the stream layout changes between the frames, change that pick's start/end in the run's picks.json")
    print(f"  and run: cut.py download --run \"{run_dir}\" --only <key>")
    for p in clean:
        print(f"  frames/{p['key']}_a.jpg  {p['key']}_b.jpg  {p['key']}_c.jpg  ({p['key']}_sheet.jpg = one frame every 2s)")


def auto_layout(run_dir, key, p=None):
    """layout.json entry from facecam.py: its look at the clip's three frames, plus one frame a second
    to find where the layout changes (segments). For Claude to check against the frames."""
    try:
        from facecam import layout_segments, suggest
        g = suggest([run_dir / "frames" / f"{key}_{t}.jpg" for t in "abc"])
        entry = {"facecam": g["facecam"], "focus_x": g["focus_x"], "mode": g["mode"], "auto_confidence": g["confidence"],
                 "note": f"auto: {g['how']}. Check the frames and fix it if it's off."}
        if p is not None:
            seq = run_dir / "frames" / f"{key}_seq"
            seq.mkdir(exist_ok=True)
            run([FFMPEG, "-v", "error", "-y", "-ss", f"{p['lead_in']:.2f}", "-t", f"{p['end_s'] - p['start_s']:.2f}",
                 "-i", str(run_dir / "raw" / f"{key}.mp4"), "-vf", "fps=1,scale=640:-2", str(seq / "%03d.jpg")], check=False)
            frames = sorted(seq.glob("*.jpg"))
            segs = layout_segments(frames, [i + 0.5 for i in range(len(frames))], g)
            shutil.rmtree(seq, ignore_errors=True)
            if segs:
                entry["segments"] = segs
                entry.update({k: segs[0][k] for k in ("facecam", "focus_x", "mode")})
                entry["note"] += " The layout changes inside the clip: see segments (seconds from the clip's start)."
        return entry
    except Exception as e:  # OpenCV missing or a bad frame: Claude fills it in by hand
        return {"facecam": None, "focus_x": 0.5, "note": f"fill in after looking at the frames (auto failed: {str(e)[:60]})"}


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


def words_for_clip(raw, lead_in, dur, model):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        run([FFMPEG, "-v", "error", "-y", "-ss", str(lead_in), "-t", str(dur), "-i", str(raw),
             "-ac", "1", "-ar", "16000", wav])
        segs, _ = model.transcribe(read_wav(wav), word_timestamps=True, vad_filter=True, beam_size=1)
        words = []
        for s in segs:
            for w in s.words or []:
                text = clean_word(w.word)
                if text:
                    words.append((max(0.0, w.start), min(dur, w.end), text))
    return words


def caption_chunks(words, max_words=3, max_chars=16):
    """Group words into short on-screen captions: [start, end, text, [(word_start, word_end, WORD), ...]]."""
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        text = " ".join(x[2] for x in cur)
        if (len(cur) >= max_words or len(text) >= max_chars or w[2][-1:] in ".!?,।" or not nxt
                or nxt[0] - w[1] > 0.6):
            shown = [(x[0], x[1], x[2].upper().strip(".,।") or x[2].upper()) for x in cur]
            chunks.append([cur[0][0], cur[-1][1], " ".join(x[2] for x in shown), shown])
            cur = []
    for i in range(len(chunks) - 1):  # hold each chunk until the next one starts (no flicker)
        if chunks[i + 1][0] - chunks[i][1] < 0.6:
            chunks[i][1] = chunks[i + 1][0]
    return chunks


def layout_lines(draw, words, size, max_w, line=1.2):
    """Wrap words into centered lines, each word in a font that can draw its script.
    Returns [(items, line_width, ascent, line_height)] with items = [(word, font, x, word_index)]."""
    from fonts import script_of
    measured = []
    for i, w in enumerate(words):
        f, rtl = font_for(w, size)
        measured.append((w, f, draw.textlength(w, font=f), rtl, i))
    space = draw.textlength(" ", font=font_for(" ", size)[0])
    rows, cur, cur_w = [], [], 0.0
    for m in measured:
        if cur and cur_w + space + m[2] > max_w:
            rows.append(cur)
            cur, cur_w = [], 0.0
        cur_w += m[2] + (space if cur else 0.0)
        cur.append(m)
    if cur:
        rows.append(cur)
    out = []
    for row in rows:
        if any(m[3] for m in row):  # Arabic, Hebrew: the first word sits on the right
            row = row[::-1]
        x, items = 0.0, []
        for m in row:
            items.append((m[0], m[1], x, m[4]))
            x += m[2] + space
        ascent = max(m[1].getmetrics()[0] for m in row)
        # Bangla, Hindi and other scripts carry marks above and below the letters: give them more room
        lh = max(size * (line if script_of(m[0])[0] is None else line * 1.25) for m in row)
        out.append((items, x - space, ascent, lh))
    return out


def draw_lines(d, lines, top, color_of, stroke=0):
    y = top
    for items, width, ascent, lh in lines:
        x0 = (OUT_W - width) / 2
        for word, f, x, i in items:
            d.text((x0 + x, y + ascent), word, font=f, fill=color_of(i), anchor="ls",
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += lh


def render_overlay(path, hook, words, active, cap_y):
    """One transparent 1080x1920 frame: the hook box and the caption, with word `active` in yellow."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if hook:
        lines = layout_lines(d, hook.upper().split(), 66, OUT_W - 160, line=1.21)
        y0 = 120
        d.rounded_rectangle([60, y0, OUT_W - 60, y0 + sum(ln[3] for ln in lines) + 40], radius=24, fill=YELLOW)
        draw_lines(d, lines, y0 + 20, lambda i: BLACK)
    if words:
        lines = layout_lines(d, list(words), 84, OUT_W - 120, line=1.19)
        draw_lines(d, lines, cap_y, lambda i: YELLOW if i == active else WHITE, stroke=8)
    img.save(path)


def build_overlay_track(tmp, hook, chunks, dur, cap_y, karaoke=True):
    """A concat list of overlay frames, one per change: hook on/off, a new caption, and with karaoke
    on, every new word, so the word being said lights up yellow."""
    def clip(t):
        return max(0.0, min(dur, t))
    times = {0.0, dur, min(HOOK_SECONDS, dur)}
    for c in chunks:
        times.update([clip(c[0]), clip(c[1])])
        if karaoke:
            times.update(clip(w[0]) for w in c[3])
    times = sorted(times)
    cache, lines = {}, []
    for a, b in zip(times, times[1:]):
        if b - a < 0.02:
            continue
        mid = (a + b) / 2
        h = hook if mid < HOOK_SECONDS else ""
        chunk = next((c for c in chunks if c[0] <= mid < c[1]), None)
        words, active = (), -1
        if chunk:
            words = tuple(w[2] for w in chunk[3])
            if karaoke:  # the last word that has started stays lit until the next one starts
                active = max((i for i, w in enumerate(chunk[3]) if w[0] <= mid), default=0)
        cy = cap_y(mid) if callable(cap_y) else cap_y
        k = (h, words, active, cy)
        if k not in cache:
            cache[k] = os.path.join(tmp, f"ov{len(cache):04d}.png")
            render_overlay(cache[k], h, words, active, cy)
        lines += [f"file '{cache[k]}'", f"duration {b - a:.3f}"]
    lines.append(lines[-2])  # concat demuxer needs the last file repeated
    lst = os.path.join(tmp, "overlay.txt")
    with open(lst, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lst


def resolve_layout(lay, facecam="on", off_mode="fit"):
    """A layout.json entry as segments [{start, end, facecam, focus_x, mode, facecam_fit}] (end None =
    to the clip's end). Modes: stack (face cam on top, gameplay under it), crop (one 9:16 crop at
    focus_x), fit (the whole picture resized to the width, blurred copy behind it). facecam="off"
    turns every stack into off_mode, so nothing is cut off the picture."""
    base = {k: v for k, v in (lay or {}).items() if k != "segments"}
    segs = lay.get("segments") if lay and lay.get("segments") else [{"start": 0.0, "end": None, **base}]
    out = []
    for i, sg in enumerate(segs):
        sg = {**base, **sg}
        mode = sg.get("mode") or ("stack" if sg.get("facecam") else "crop")
        if mode == "stack" and not sg.get("facecam"):
            mode = "crop"
        if facecam == "off" and mode == "stack":
            mode = off_mode
        end = sg.get("end") if i + 1 < len(segs) else None
        out.append({"start": float(sg.get("start") or 0.0), "end": None if end is None else float(end),
                    "facecam": sg.get("facecam"), "focus_x": sg.get("focus_x"), "mode": mode,
                    "facecam_fit": sg.get("facecam_fit", "auto")})
    return out


def _even(x):
    return max(2, int(x) // 2 * 2)


def segment_filter(inp, out, src_w, src_h, sg, tag):
    """ffmpeg filter for one layout segment, and where its captions go (y in pixels)."""
    fx = sg.get("focus_x")
    fx = min(1.0, max(0.0, 0.5 if fx is None else float(fx)))  # 0 is a real value: the far left
    blur = "boxblur=24:4,eq=brightness=-0.12"
    if sg["mode"] == "stack":
        x, y, w, h = [min(1.0, max(0.0, float(v))) for v in sg["facecam"]]
        x, y = min(x, 0.95), min(y, 0.95)
        cx, cy = int(x * src_w), int(y * src_h)
        cw, ch = max(16, int(w * src_w)), max(16, int(h * src_h))
        cw, ch = min(cw, src_w - cx), min(ch, src_h - cy)
        gw = min(src_w, int(src_h * OUT_W / (OUT_H - TOP_H)) // 2 * 2)
        gx = int(min(max(0, fx * src_w - gw / 2), src_w - gw))
        shape = (cw / ch) / (OUT_W / TOP_H)
        # a 16:9 webcam loses a little off its sides and fills the band; a much wider (or taller) box,
        # like two cut-out people side by side, is shown whole over a blurred copy so nobody is cut off
        fit = sg.get("facecam_fit") == "fit" or (sg.get("facecam_fit", "auto") == "auto" and not 0.77 <= shape <= 1.3)
        crop = f"crop={cw}:{ch}:{cx}:{cy}"
        if fit:  # the whole face cam, nobody cut off the sides; a blurred copy fills the band
            top = (f"[{tag}a]{crop},scale={OUT_W}:{TOP_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{TOP_H},{blur}[{tag}tb];"
                   f"[{tag}b]{crop},scale={OUT_W}:{TOP_H}:force_original_aspect_ratio=decrease:force_divisible_by=2[{tag}tf];"
                   f"[{tag}tb][{tag}tf]overlay=(W-w)/2:(H-h)/2[{tag}top];")
            split = f"{inp}split=3[{tag}a][{tag}b][{tag}c];"
        else:
            top = (f"[{tag}a]{crop},scale={OUT_W}:{TOP_H}:force_original_aspect_ratio=increase,"
                   f"crop={OUT_W}:{TOP_H}[{tag}top];")
            split = f"{inp}split=2[{tag}a][{tag}c];"
        return (split + top + f"[{tag}c]crop={gw}:{src_h}:{gx}:0,scale={OUT_W}:{OUT_H - TOP_H}[{tag}bot];"
                f"[{tag}top][{tag}bot]vstack=inputs=2,setsar=1{out}"), TOP_H + 70
    if sg["mode"] == "fit":
        fg_h = _even(OUT_W * src_h / src_w)
        return (f"{inp}split=2[{tag}a][{tag}b];"
                f"[{tag}a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},{blur}[{tag}bg];"
                f"[{tag}b]scale={OUT_W}:{fg_h}[{tag}fg];[{tag}bg][{tag}fg]overlay=0:(H-h)/2,setsar=1{out}"), \
            int((OUT_H + fg_h) / 2 + 60)
    gw = min(src_w, int(src_h * OUT_W / OUT_H) // 2 * 2)
    gx = int(min(max(0, fx * src_w - gw / 2), src_w - gw))
    return f"{inp}crop={gw}:{src_h}:{gx}:0,scale={OUT_W}:{OUT_H},setsar=1{out}", 1250


def layout_filter(src_w, src_h, lay, dur=None, facecam="on", off_mode="fit"):
    """The whole clip's video filter (ending in [base]) and the caption height: a number, or a
    function of time when the clip has segments with different layouts."""
    segs = resolve_layout(lay, facecam, off_mode)
    if len(segs) == 1:
        return segment_filter("[0:v]", "[base]", src_w, src_h, segs[0], "s0")
    parts = [f"[0:v]split={len(segs)}" + "".join(f"[in{i}]" for i in range(len(segs)))]
    caps = []
    for i, sg in enumerate(segs):
        a, b = sg["start"], sg["end"] if sg["end"] is not None else (dur or 1e9)
        parts.append(f"[in{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[t{i}]")
        f, cap = segment_filter(f"[t{i}]", f"[s{i}]", src_w, src_h, sg, f"s{i}")
        parts.append(f)
        caps.append((a, b, cap))
    parts.append("".join(f"[s{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0,setsar=1[base]")
    return ";".join(parts), (lambda t: next((c for a, b, c in caps if a <= t < b), caps[-1][2]))


def segment_at(segs, t):
    return next((sg for sg in segs if sg["start"] <= t < (sg["end"] if sg["end"] is not None else 1e9)), segs[-1])


def frame_brightness(path, t, box):
    from PIL import Image, ImageStat
    with tempfile.TemporaryDirectory() as tmp:
        jpg = os.path.join(tmp, "f.jpg")
        run([FFMPEG, "-v", "error", "-y", "-ss", str(t), "-i", str(path), "-frames:v", "1", jpg], check=False)
        if not os.path.exists(jpg):
            return 0.0
        return sum(ImageStat.Stat(Image.open(jpg).convert("L").crop(box)).mean)


def render_one(p, raw, out, dur, lay, model, venc, karaoke=True, facecam="on", off_mode="fit"):
    words = words_for_clip(raw, p["lead_in"], dur, model)
    chunks = caption_chunks(words)
    src_w, src_h = video_size(raw)
    vf, cap_y = layout_filter(src_w, src_h, lay, dur, facecam, off_mode)
    segs = resolve_layout(lay, facecam, off_mode)
    with tempfile.TemporaryDirectory() as tmp:
        lst = build_overlay_track(tmp, p.get("hook", ""), chunks, dur, cap_y, karaoke)
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
    sg = segment_at(segs, t_check)
    if sg["mode"] == "stack":
        if frame_brightness(out, t_check, (0, 0, OUT_W, TOP_H)) < 12:
            problems.append("face cam area looks black, check the facecam box in layout.json")
        game_box = (0, TOP_H + 420, OUT_W, OUT_H)
    elif sg["mode"] == "fit":
        game_box = (0, OUT_H // 2 - 200, OUT_W, OUT_H // 2 + 200)
    else:
        game_box = (0, 360, OUT_W, 1200)
    if frame_brightness(out, t_check, game_box) < 6:
        problems.append("gameplay area looks black")
    if not words:
        problems.append("no speech found, so no captions (fine for a pure reaction scream)")
    return problems, got, words


# ---------------------------------------------------------------- render

def moment_signals(p):
    """The analyzer's signals at the candidate moment this clip came from, for learn.py to learn
    which signals your audience actually rewards. Empty if the clip wasn't near a candidate."""
    doc = load(WORK / "streams" / str(p.get("source")) / "candidates.json", {}) or {}
    near = [c for c in doc.get("candidates", []) if p["start_s"] - 5 <= c.get("moment", -1e9) <= p["end_s"] + 5]
    if not near:
        return {}
    best = max(near, key=lambda c: c.get("strength", 0))
    return {**best.get("signals", {}), "strength": best.get("strength", 0)}


def cmd_render(a):
    run_dir = os.path.expanduser(a.run)
    doc = load(os.path.join(run_dir, "picks.json"))
    if not doc:
        sys.exit(f"No picks.json in {run_dir}. Use the run folder that cut.py download printed.")
    picks = doc["picks"]
    only = {k.zfill(2) for k in a.only} if a.only else None
    layout = load(os.path.join(run_dir, "layout.json"), {})
    out_dir = os.path.join(run_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    model = load_whisper(a.model)
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
        if p.get("status") == "download failed":
            print(f"!! {key}: never downloaded. Retry: cut.py download --run \"{run_dir}\" --only {key}")
            continue
        if only and key not in only:
            if key in previous:
                results.append(previous[key])
            continue
        raw = os.path.join(run_dir, "raw", f"{key}.mp4")
        dur = p["end_s"] - p["start_s"]
        lay = layout.get(key, {})
        name = f"{key}_{slug(p.get('hook') or p.get('title') or key)}.mp4"
        out = os.path.join(out_dir, name)
        modes = [sg["mode"] for sg in resolve_layout(lay, a.facecam, a.off_mode)]
        names = {"stack": "face cam on top", "crop": "full-frame crop", "fit": "whole picture, resized"}
        print(f"==> {key}: rendering {dur:.0f}s clip ({' then '.join(names[m] for m in modes)})")
        if not os.path.exists(raw):
            print(f"!! {key}: no downloaded video at {raw}. Run cut.py download again for it. Skipping.")
            continue
        try:
            problems, got, words = render_one(p, raw, out, dur, lay, model, venc, a.captions == "karaoke",
                                              a.facecam, a.off_mode)
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
    header = ["date", "run", "file", "source_url", "start", "streamer", "moment_type", "score", "hook", "title", "signals"]
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
                                      p.get("hook", ""), p.get("title", ""), json.dumps(moment_signals(p))])))
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
    r.add_argument("--captions", choices=["karaoke", "plain"], default="karaoke",
                   help="karaoke lights up each word in yellow as it's said; plain keeps every word white")
    r.add_argument("--facecam", choices=["on", "off"], default="on",
                   help="on: use the face cam boxes in layout.json; off: no face cam band, the whole picture "
                        "is resized to fit (or cropped, with --off-mode crop)")
    r.add_argument("--off-mode", choices=["fit", "crop"], default="fit",
                   help="with --facecam off: fit shows the whole picture with a blurred fill, crop fills the screen")
    r.add_argument("--delete-raw", action="store_true", help="delete the downloaded source minutes afterwards")
    a = ap.parse_args()
    cmd_download(a) if a.cmd == "download" else cmd_render(a)


if __name__ == "__main__":
    main()
