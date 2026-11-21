"""Look at what's working: match your posted Shorts to the clips this skill made.

Reads the public view counts on your YouTube Shorts tab (no API key needed),
matches each Short to log.csv by title, and reports which moment types and
streamers beat your average.

Usage: python learn.py --channel https://www.youtube.com/@yourhandle
"""
import argparse
import csv
import difflib
import json
import math
import statistics
from collections import defaultdict

import yt_dlp

from common import WORK, explain_download_error, save, ydl_opts


def learn_weights(matched, min_clips=10):
    """Which analyzer signals go with more views on this channel. For each signal, the correlation
    between its value at the clip's moment and log(views) becomes a gentle multiplier on that
    signal's weight (0.6x to 1.6x), so the next analysis leans toward what this audience watches.
    Returns ({signal: multiplier}, {signal: correlation}) or (None, None) with too little data."""
    rows = []
    for m in matched:
        try:
            sig = json.loads(m.get("signals") or "{}")
        except ValueError:
            continue
        if sig:
            rows.append((sig, math.log1p(m["views"])))
    if len(rows) < min_clips:
        return None, None
    keys = sorted({k for sig, _ in rows for k in sig if k != "strength"})
    weights, corr = {}, {}
    ys = [y for _, y in rows]
    for k in keys:
        xs = [sig.get(k, 0.0) for sig, _ in rows]
        if len(set(xs)) < 3:
            continue
        r = statistics.correlation(xs, ys)
        corr[k] = round(r, 2)
        weights[k] = round(min(1.6, max(0.6, math.exp(0.7 * r))), 2)
    return weights, corr


def norm(s):
    return " ".join(str(s or "").lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, help="your channel URL, e.g. https://www.youtube.com/@name")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    log_path = WORK / "log.csv"
    if not log_path.exists():
        raise SystemExit("No log.csv yet. Make and post some clips first.")
    with open(log_path, newline="", encoding="utf-8") as fh:
        made = list(csv.DictReader(fh))

    ch = a.channel.strip().rstrip("/")
    if ch.startswith("@"):
        ch = "https://www.youtube.com/" + ch
    elif "://" not in ch:
        ch = "https://" + ch
    for tab in ("/shorts", "/videos", "/featured", "/streams"):
        if ch.endswith(tab):
            ch = ch[: -len(tab)]
    url = ch + "/shorts"
    try:
        with yt_dlp.YoutubeDL(ydl_opts(extract_flat="in_playlist", playlistend=a.limit)) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        low = str(e).lower()
        if "does not have a shorts tab" in low:
            raise SystemExit("That channel has no Shorts yet. Post a few clips first, then run this again.")
        if "404" in low or "does not exist" in low:
            raise SystemExit(f"Couldn't find a YouTube channel at {ch}. Use your channel link, "
                             "e.g. https://www.youtube.com/@yourhandle")
        raise SystemExit(explain_download_error(e))
    posted = [e for e in (info.get("entries") or []) if e and e.get("title")]
    if not posted:
        raise SystemExit("Couldn't read any Shorts from that channel.")

    # A Short matches a clip when its title contains the clip's title or hook (small edits like
    # emojis or "#shorts" are fine), or is almost identical to one of them. Loose matching would
    # pair other people's Shorts with your clips and give you wrong numbers.
    matched, used = [], set()
    for e in posted:
        pt = norm(e["title"])
        best, score = None, 0.0
        for i, m in enumerate(made):
            for field in ("title", "hook"):
                f = norm(m.get(field))
                if len(f) < 8:
                    continue
                r = 1.0 if f in pt else difflib.SequenceMatcher(None, pt, f).ratio()
                if r > score:
                    best, score = (i, m), r
        if best and score >= 0.9 and best[0] not in used:
            used.add(best[0])
            matched.append({**best[1], "views": e.get("view_count") or 0, "posted_title": e["title"], "url": e.get("url")})

    if not matched:
        print(f"Read {len(posted)} Shorts, but none matched the clips in log.csv. "
              "Keep the clip title (or hook) in your post title so they can be matched.")
        return

    if len(matched) < 10:
        print(f"Only {len(matched)} of your Shorts matched clips from this skill. That's not enough to see a pattern yet "
              "(it needs about 10). Keep posting, and keep the clip's title or hook in the post title.")
    views = [m["views"] for m in matched]
    avg = statistics.mean(views)
    out = [f"# What's working ({len(matched)} of your Shorts matched)", f"Average views: {avg:,.0f}", ""]

    for field, label in (("moment_type", "Moment type"), ("streamer", "Streamer")):
        groups = defaultdict(list)
        for m in matched:
            groups[m.get(field) or "unknown"].append(m["views"])
        out.append(f"## {label}")
        for k, v in sorted(groups.items(), key=lambda kv: -statistics.mean(kv[1])):
            mean = statistics.mean(v)
            out.append(f"- {k}: {mean:,.0f} avg views over {len(v)} clips ({mean / avg:.1f}x your average)")
        out.append("")

    ranked = sorted(matched, key=lambda m: -m["views"])
    out.append("## Top 5")
    out += [f"- {m['views']:,} views | {m.get('moment_type')} | {m.get('streamer')} | hook: {m.get('hook')}" for m in ranked[:5]]
    out.append("\n## Bottom 5")
    out += [f"- {m['views']:,} views | {m.get('moment_type')} | {m.get('streamer')} | hook: {m.get('hook')}" for m in ranked[-5:]]

    weights, corr = learn_weights(matched)
    if weights:
        save(WORK / "weights.json", weights)
        out.append("\n## Signal weights learned from your views (used by the next analysis)")
        out += [f"- {k}: x{weights[k]} (correlation with views {corr[k]:+.2f})" for k in sorted(weights, key=lambda k: -weights[k])]
    else:
        out.append("\n(Signal weights need 10+ matched clips made with the new analyzer; keeping the defaults.)")

    report = "\n".join(out)
    (WORK / "learn_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {WORK / 'learn_report.md'}. Claude: update {WORK / 'preferences.md'} from this.")


if __name__ == "__main__":
    main()
