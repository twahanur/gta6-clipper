"""Shared helpers for the Clipper scripts."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

# Keep Whisper's engine (ctranslate2) from printing its own tips and warnings.
os.environ.setdefault("CT2_VERBOSE", "-1")

HOME_DIR = pathlib.Path.home() / ".clipper"
WORK = pathlib.Path(os.environ.get("CLIPPER_DIR") or os.environ.get("GTA6_CLIPPER_DIR") or (pathlib.Path.home() / "Clipper"))

# Make deno (YouTube JS challenges) and uv visible to yt-dlp.
os.environ["PATH"] = os.pathsep.join(
    # ~/.clipper/bin holds our ffmpeg; yt-dlp's partial (clip-only) downloads look for it on PATH
    [str(HOME_DIR / "bin"), str(HOME_DIR / "venv" / "bin"), str(pathlib.Path.home() / ".deno" / "bin"), str(pathlib.Path.home() / ".local" / "bin"),
     os.environ.get("PATH", "")]
)


def ffmpeg_path():
    p = HOME_DIR / "bin" / "ffmpeg"
    if p.exists():
        return str(p)
    found = shutil.which("ffmpeg")
    if found:
        return found
    sys.exit("ffmpeg not found. Run scripts/setup.sh first.")


FFMPEG = ffmpeg_path()


def run(cmd, check=True, capture=False):
    r = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        tail = (r.stderr or "")[-1500:]
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:6]))} ...\n{tail}")
    return r


class _QuietLogger:
    """yt-dlp's own errors are raised as exceptions and explained by explain_download_error(),
    so don't also print its raw multi-line messages."""
    def debug(self, msg):
        pass

    info = warning = debug

    def error(self, msg):
        pass


def ydl_opts(**extra):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
        "ffmpeg_location": FFMPEG,
    }
    browser = os.environ.get("CLIPPER_COOKIES_BROWSER") or os.environ.get("GTA6_COOKIES_BROWSER")  # e.g. chrome, safari, firefox (only if YouTube asks you to sign in)
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    opts.update(extra)
    return opts


def hms(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_time(t):
    """Accepts seconds (number or '93.5') or 'HH:MM:SS' / 'MM:SS'."""
    if isinstance(t, (int, float)):
        return float(t)
    t = str(t).strip()
    if re.fullmatch(r"[\d.]+", t):
        return float(t)
    parts = [float(p) for p in t.split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def slug(s, n=40):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:n] or "clip"


def load(path, default=None):
    path = pathlib.Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


VIDEOS = WORK / "videos"  # every finished clip, numbered 001, 002, ... across all runs


def next_serial(folder):
    nums = [int(m.group(1)) for n in (os.listdir(folder) if os.path.isdir(folder) else [])
            if (m := re.match(r"(\d{3,})_.*\.mp4$", n))]
    return max(nums, default=0) + 1


def publish_video(clip, run_dir, key, name_slug, folder=None):
    """Copy a finished clip to the videos folder as NNN_<slug>.mp4. A re-rendered clip keeps its number."""
    folder = str(folder or VIDEOS)
    os.makedirs(folder, exist_ok=True)
    book_path = os.path.join(run_dir, "videos.json")
    book = load(book_path, {}) or {}
    serial = book.get(key)
    if serial is None:
        serial = next_serial(folder)
    else:
        for n in os.listdir(folder):
            if n.startswith(f"{serial:03d}_") and n.endswith(".mp4"):
                os.remove(os.path.join(folder, n))
    dest = os.path.join(folder, f"{serial:03d}_{name_slug}.mp4")
    shutil.copy2(clip, dest)
    book[key] = serial
    save(book_path, book)
    return dest


def stream_dir(video_id):
    d = WORK / "streams" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def media_duration(path):
    r = run([FFMPEG, "-hide_banner", "-i", str(path)], check=False)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr or "")
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def read_wav(path):
    """Load a 16 kHz mono 16-bit wav as float32 for Whisper.
    Passing an array (not a path) means Whisper never touches PyAV, whose newer
    versions break faster-whisper's own decoder ('metadata_errors' TypeError)."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def explain_download_error(e):
    """Turn a yt-dlp error into one plain sentence the user can act on."""
    msg = str(e)
    low = msg.lower()
    if "confirm your age" in low or "age-restricted" in low:
        return ("This video is age-restricted, so YouTube wants you signed in. Run "
                "`export CLIPPER_COOKIES_BROWSER=chrome` (or safari/firefox, whichever browser you're logged into "
                "YouTube with) and try again, or pick a different stream.")
    if "not a bot" in low or "sign in" in low or "http error 403" in low:
        return ("YouTube is blocking the download. Run setup again to update the downloader, and if it still "
                "fails, run `export CLIPPER_COOKIES_BROWSER=chrome` (or safari/firefox) and try again.")
    if "private video" in low or "members-only" in low or "join this channel" in low:
        return "This video is private or members-only, so it can't be clipped. Pick a different stream."
    if "unavailable" in low or "removed" in low or "does not exist" in low:
        return "This video isn't available anymore (removed, blocked in your country, or a wrong link)."
    first = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?", "", msg.splitlines()[0] if msg else "")
    return "Download failed: " + first[:300]
