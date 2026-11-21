"""See what happens on screen, cheaply (analyze.py --visual).

Downloads a 144p copy of the stream (about the size of the audio), looks at 4 frames a second
shrunk to 64x36 pixels, and gives one value per second for:
  motion   bursts of the picture changing (chases, crashes, the streamer jumping out of the chair)
  gray     the picture losing its color all at once: death and fail screens ("WASTED", "MISSION
           FAILED"), but also cutscene fades and loading screens, so it's only a hint
"""
import subprocess

import numpy as np
import yt_dlp

from common import FFMPEG, ydl_opts
from hype import onset, robust_z, rolling_median, smooth

FPS = 4
W, H = 64, 36


def download_low_video(url, d):
    for f in d.glob("video_low.*"):
        if f.suffix not in (".part", ".ytdl"):
            return f
    # HLS formats download in small pieces: fetch 8 at once, or an 11-hour stream takes 20 minutes
    opts = ydl_opts(format="269/160/603/602/worstvideo[vcodec^=avc1]/worstvideo/worst", outtmpl=str(d / "video_low.%(ext)s"),
                    concurrent_fragment_downloads=8)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return next(f for f in d.glob("video_low.*") if f.suffix not in (".part", ".ytdl"))


def frame_stats(frames):
    """[k, H, W, 3] uint8 frames -> per-frame brightness, colorfulness, and the gray image."""
    f = frames.astype(np.float32) / 255.0
    mx, mn = f.max(axis=3), f.min(axis=3)
    gray = f.mean(axis=3)
    sat = np.where(mx > 0.08, (mx - mn) / (mx + 1e-6), 0.0)  # near-black pixels have no color to lose
    return gray.mean(axis=(1, 2)), sat.mean(axis=(1, 2)), gray


def visual_features(video, n):
    """Per-second motion, brightness and colorfulness for n seconds of video."""
    size = W * H * 3
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(video), "-vf", f"fps={FPS},scale={W}:{H}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    bright, sat, motion = [], [], []
    prev = None
    while True:
        buf = p.stdout.read(size * FPS * 300)  # five minutes at a time
        k = len(buf) // size
        if k == 0:
            break
        frames = np.frombuffer(buf[: k * size], dtype=np.uint8).reshape(k, H, W, 3)
        b, s, g = frame_stats(frames)
        g_prev = np.concatenate([g[:1] if prev is None else prev[None], g[:-1]])
        motion.append(np.abs(g - g_prev).mean(axis=(1, 2)))
        bright.append(b)
        sat.append(s)
        prev = g[-1]
    p.wait()
    if not bright:
        return None
    per = lambda x, fn: fn(np.pad(x, (0, (-len(x)) % FPS), mode="edge").reshape(-1, FPS), axis=1)
    bright, sat, motion = (np.concatenate(x) for x in (bright, sat, motion))
    out = {"motion": per(motion, np.mean), "bright_max": per(bright, np.max), "bright": per(bright, np.mean),
           "sat": per(sat, np.mean)}
    return {k: np.pad(v[:n], (0, max(0, n - len(v))), mode="edge") for k, v in out.items()}


def gray_events(sat, drop=0.45, hold=2):
    """Seconds where the picture suddenly lost most of its color compared to the minute before."""
    before = rolling_median(np.concatenate([np.full(60, sat[0]), sat])[:len(sat)], 61)
    low = (sat < drop * before) & (before > 0.12)
    ev = np.zeros(len(sat))
    run = 0
    for t, is_low in enumerate(low):
        run = run + 1 if is_low else 0
        if run == hold:
            ev[max(0, t - hold + 1)] = 1.0
    return ev


def visual_signal(vf):
    """One per-second 'something is happening on screen' value, and a hint label per second.

    Measured on an 11-hour GTA stream: motion bursts were 1.36x more common at the real moments
    (found by chat and voice) than at random seconds, so motion counts. Flashes (1.06x) and color
    draining away (no better than chance: cutscene fades, menus and loading screens look the same
    as death screens) don't, so the gray screen is only a hint for Claude, never a score."""
    motion = np.clip(robust_z(smooth(onset(vf["motion"], 30), 2)), 0, None)
    gray = gray_events(vf["sat"])
    labels = [""] * len(motion)
    for t in np.where(gray > 0)[0]:
        for u in range(int(t), min(len(labels), int(t) + 6)):
            labels[u] = "screen went dark or gray (fail screen, cutscene or loading?)"
    for t in np.where(motion > 4)[0]:
        if not labels[t]:
            labels[t] = "big on-screen motion"
    return smooth(np.minimum(motion, 6), 3), labels


# ---------------------------------------------------------------- the streamer moving in the face cam

def find_stream_facecam(url, d, duration, samples=5):
    """Face cam box for the whole stream: grab 2 seconds at 720p from a few points, one frame each,
    and run facecam.py on them together. Cached in facecam.json. None if there's no face cam."""
    import json
    from yt_dlp.utils import download_range_func
    from facecam import suggest
    cache = d / "facecam.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("facecam")
    fdir = d / "facecam_frames"
    fdir.mkdir(exist_ok=True)
    frames = []
    for i in range(samples):
        t = duration * (i + 1) / (samples + 1)
        out = fdir / f"s{i}.mp4"
        opts = ydl_opts(format="bv*[height<=720][vcodec^=avc1]/bv*[height<=720]/b[height<=720]/b",
                        outtmpl=str(out), download_ranges=download_range_func(None, [(t, t + 2)]),
                        force_keyframes_at_cuts=False)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            jpg = fdir / f"s{i}.jpg"
            subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(out), "-frames:v", "1", str(jpg)], check=False)
            if jpg.exists():
                frames.append(jpg)
        except Exception as e:
            print(f"   (face cam sample {i + 1} failed: {str(e)[:60]})")
    g = suggest(frames) if len(frames) >= 2 else {"facecam": None, "how": "not enough frames"}
    cache.write_text(json.dumps(g))
    return g.get("facecam")


def facecam_motion(video, box, n):
    """Per-second motion inside the face cam box: the streamer jumping, waving, grabbing their head."""
    x, y, w, h = box
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(video), "-vf",
                          f"fps={FPS},crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale=32:24", "-pix_fmt", "gray",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = 32 * 24
    out, prev = [], None
    while True:
        buf = p.stdout.read(size * FPS * 300)
        k = len(buf) // size
        if k == 0:
            break
        g = np.frombuffer(buf[: k * size], dtype=np.uint8).reshape(k, 24, 32).astype(np.float32) / 255
        g_prev = np.concatenate([g[:1] if prev is None else prev[None], g[:-1]])
        out.append(np.abs(g - g_prev).mean(axis=(1, 2)))
        prev = g[-1]
    p.wait()
    if not out:
        return None
    m = np.concatenate(out)
    m = np.pad(m, (0, (-len(m)) % FPS), mode="edge").reshape(-1, FPS).mean(axis=1)
    return np.pad(m[:n], (0, max(0, n - len(m))), mode="edge")
