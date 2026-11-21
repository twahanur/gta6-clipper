"""Find the moments other clippers already posted from this stream, and how many views they got.

  python fingerprint.py --stream <id> [--query "kai cenat gta 5"] [--max 40]

Searches YouTube for short videos about the stream (posted after it), downloads only their audio,
and finds where each one sits in the stream by matching sound "fingerprints": the shape of the
spectrum 10 times a second, compared with cross-correlation. A match gives the exact second in the
stream and the Short's views. analyze.py uses these as a signal ("already_clipped") and as proof
in for_claude.md, and benchmark.py uses them as ground truth.

Writes ~/Clipper/streams/<id>/already_clipped.json.
"""
import argparse
import pathlib
import subprocess
import tempfile
import urllib.parse

import numpy as np
import yt_dlp

from common import FFMPEG, WORK, explain_download_error, hms, load, media_duration, save, ydl_opts

SR = 8000
HOP = 800          # 10 frames a second
NFFT = 2048
BANDS = 24


def _band_edges():
    f = np.geomspace(150, 3800, BANDS + 1)
    return np.searchsorted(np.fft.rfftfreq(NFFT, 1 / SR), f)


EDGES = _band_edges()


def prints_from_pcm(x):
    """[frames, BANDS] fingerprint: log band energies, each frame minus its own mean (spectral shape,
    so volume changes and a different loudness on the Short don't matter), then each band normalized."""
    n = (len(x) - NFFT) // HOP
    if n <= 0:
        return np.zeros((0, BANDS), dtype=np.float32)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(NFFT), axis=1)) ** 2
    e = np.log(np.stack([spec[:, a:b].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])], axis=1) + 1e-6)
    e -= e.mean(axis=1, keepdims=True)
    e = np.diff(e, axis=0, prepend=e[:1])  # how the shape changes: robust to music beds and EQ
    return ((e - e.mean(axis=0)) / (e.std(axis=0) + 1e-6)).astype(np.float32)


def audio_prints(path, chunk_s=600):
    """Fingerprint of a whole file, streamed in 10-minute pieces so 11 hours never sit in memory."""
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE)
    parts, carry = [], np.zeros(0, dtype=np.float32)
    while True:
        buf = p.stdout.read(SR * chunk_s * 4)
        if not buf:
            break
        x = np.concatenate([carry, np.frombuffer(buf[: len(buf) // 4 * 4], dtype=np.float32)])
        n = (len(x) - NFFT) // HOP
        if n > 0:
            parts.append(prints_from_pcm(x[: n * HOP + NFFT]))
            carry = x[n * HOP:]
        else:
            carry = x
    p.wait()
    return np.concatenate(parts) if parts else np.zeros((0, BANDS), dtype=np.float32)


def stream_prints(d):
    cache = d / "fingerprint.npy"
    if cache.exists():
        return np.load(cache)
    audio = next(f for f in d.glob("audio.*") if f.suffix not in (".part", ".ytdl"))
    fp = audio_prints(audio)
    np.save(cache, fp)
    return fp


class Matcher:
    """Finds where clips sit in one stream. The stream's FFT is computed once and reused per clip."""

    def __init__(self, stream_fp, piece=120):
        self.n, self.piece = len(stream_fp), piece
        self.size = 1 << int(np.ceil(np.log2(self.n + piece)))
        self.S = np.fft.rfft(stream_fp, self.size, axis=0)

    def locate(self, clip_fp):
        """(seconds into the stream, confidence) for a clip. Tries its start, middle and end
        separately, because Shorts are often cut together from pieces."""
        best = (None, 0.0)
        if len(clip_fp) < 40:
            return best
        p = self.piece
        for s0 in sorted({0, max(0, len(clip_fp) // 2 - p // 2), max(0, len(clip_fp) - p)}):
            q = clip_fp[s0:s0 + p]
            if len(q) < 40:
                continue
            Q = np.fft.rfft(q[::-1], self.size, axis=0)
            xc = np.fft.irfft(self.S * Q, self.size, axis=0).sum(axis=1)[len(q) - 1:self.n]
            if not len(xc):
                continue
            k = int(np.argmax(xc))
            conf = float((xc[k] - np.median(xc)) / (xc.std() + 1e-6))
            if conf > best[1]:
                best = ((k - s0) * HOP / SR, conf)
        return best


def search_shorts(query, limit):
    opts = ydl_opts(extract_flat="in_playlist", playlistend=limit)
    with yt_dlp.YoutubeDL(opts) as ydl:
        # YouTube search, "under 4 minutes" filter
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query) + "&sp=EgIYAQ%253D%253D"
        res = ydl.extract_info(url, download=False)
    out = []
    for e in res.get("entries") or []:
        dur = (e or {}).get("duration") or 0
        if e and e.get("id") and (dur == 0 or 8 <= dur <= 180):  # Shorts in search results often have no duration
            out.append({"id": e["id"], "url": f"https://www.youtube.com/watch?v={e['id']}", "title": e.get("title"),
                        "channel": e.get("channel") or e.get("uploader"), "views": e.get("view_count") or 0,
                        "duration": dur})
    return out


def clip_audio(url, tmp):
    opts = ydl_opts(format="bestaudio/worst", outtmpl=str(pathlib.Path(tmp) / "%(id)s.%(ext)s"))
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return pathlib.Path(ydl.prepare_filename(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, help="stream id (folder name in ~/Clipper/streams)")
    ap.add_argument("--query", nargs="*", help="what to search for, one or more searches (default: channel name and title words)")
    ap.add_argument("--max", type=int, default=40, help="how many search results to check")
    ap.add_argument("--min-confidence", type=float, default=15.0,
                    help="below ~15, matches are often a game cutscene that also plays in someone else's video")
    a = ap.parse_args()
    d = WORK / "streams" / a.stream
    meta = load(d / "meta.json")
    if not meta:
        raise SystemExit(f"No analyzed stream at {d}. Run analyze.py on it first.")
    words = [w for w in (meta.get("title") or "").replace("🎮", " ").split() if w.isalnum() and len(w) > 2][:4]
    queries = a.query or [f"{meta.get('channel', '').replace(' Live', '')} {' '.join(words)} shorts"]
    print("==> Fingerprinting the stream's audio (once, cached)")
    matcher = Matcher(stream_prints(d))
    shorts, seen = [], set()
    for query in queries:
        print(f"==> Searching YouTube: {query}")
        try:
            for sh in search_shorts(query, a.max):
                if sh["id"] not in seen:
                    seen.add(sh["id"])
                    shorts.append(sh)
        except yt_dlp.utils.DownloadError as e:
            print(f"   search failed: {explain_download_error(e)}")
    print(f"   {len(shorts)} short videos to check")
    found = []
    with tempfile.TemporaryDirectory() as tmp:
        for sh in shorts:
            try:
                path = clip_audio(sh["url"], tmp)
            except Exception as e:
                print(f"   skip {sh['id']}: {str(e)[:60]}")
                continue
            t, conf = matcher.locate(audio_prints(path))
            if t is not None and conf >= a.min_confidence and 0 <= t <= meta["duration"]:
                info_dur = sh["duration"] or media_duration(path)
                found.append({**sh, "duration": info_dur, "offset": round(t, 1), "confidence": round(conf, 1)})
                print(f"   MATCH {hms(t)}  {sh['views']:,} views  {sh['title'][:60]}  (confidence {conf:.0f})")
            path.unlink(missing_ok=True)
    found.sort(key=lambda f: -f["views"])
    save(d / "already_clipped.json", found)
    print(f"\n{len(found)} of {len(shorts)} short videos came from this stream. Saved to {d / 'already_clipped.json'}")


if __name__ == "__main__":
    main()
