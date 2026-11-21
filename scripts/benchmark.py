"""Measure how well the analyzer finds the moments people actually liked.

  python benchmark.py --stream <id> [--k 10 20 40] [--tolerance 30]

Ground truth comes from whatever the stream folder has:
  labels.json           your own list of good moments: ["01:23:45", 5025, ...]
  community_clips.json  Twitch clips viewers made from the VOD (twitch.py)
  already_clipped.json  Shorts other clippers posted from it, found by fingerprint.py
  comments.json         timestamps viewers left in the comments
For each source it scores the stream from the saved signals (no transcribing, a few seconds), leaving
out the signal made from that same source so it can't grade its own homework, and reports:
  precision@K  how many of the top K moments are within --tolerance seconds of a real one
  recall@K     how many real moments the top K found
  baseline     what random picks would score, for comparison
then removes each signal in turn to show what it adds. Results are appended to ~/Clipper/benchmark.csv.
"""
import argparse
import csv
import datetime as dt

import numpy as np

import hype
from analyze import collect_signals
from common import WORK, load, parse_time

LEAK = {"community_clips": "community_clips", "already_clipped": "already_clipped", "bookmarks": "viewer_bookmarks"}


def load_features(d):
    z = np.load(d / "features.npz", allow_pickle=False)
    return {k: z[k] for k in z.files if k not in ("version", "model", "labels")}


def label_sources(d, ex, n):
    out = {}
    manual = load(d / "labels.json", None)
    if manual:
        out["manual"] = [parse_time(t) for t in manual]
    clips = load(d / "community_clips.json", None)
    if clips:
        out["community_clips"] = [c["offset"] + min(10, c.get("duration", 30) / 2) for c in clips]
    already = [x for x in (load(d / "already_clipped.json", []) or []) if x.get("confidence", 99) >= 15]
    if len(already) >= 3:
        out["already_clipped"] = [c["offset"] + min(10, c.get("duration", 30) / 2) for c in already]
    if ex["marks"]:
        out["bookmarks"] = [t for t, _, _ in ex["marks"]]
    return {k: dedupe([t for t in v if 0 <= t < n]) for k, v in out.items()}


def dedupe(times, gap=60):
    out = []
    for t in sorted(times):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def evaluate(score, labels, k, tol, spacing):
    moments = hype.pick_peaks(score, k, spacing)
    lab = np.array(labels)
    hit = sum(1 for m in moments if np.min(np.abs(lab - m)) <= tol)
    found = sum(1 for t in labels if any(abs(t - m) <= tol for m in moments))
    return hit / max(1, len(moments)), found / max(1, len(labels))


def baseline(labels, n, k, tol):
    covered = np.zeros(n, dtype=bool)
    for t in labels:
        covered[max(0, int(t - tol)):min(n, int(t + tol) + 1)] = True
    return float(covered.mean()), min(1.0, k * (2 * tol + 1) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--k", type=int, nargs="*", default=[10, 20, 40])
    ap.add_argument("--tolerance", type=int, default=30)
    a = ap.parse_args()
    d = WORK / "streams" / a.stream
    if not (d / "features.npz").exists():
        raise SystemExit(f"Run analyze.py on this stream first ({d} has no features.npz).")
    feat = load_features(d)
    n = len(feat["loud"])
    signals, ex = collect_signals(d, feat, use_visual=(d / "visual.npz").exists(), quiet=True)
    weights = load(WORK / "weights.json", None)
    spacing = 30 if n < 1800 else 90
    sources = label_sources(d, ex, n)
    if not sources:
        raise SystemExit("No ground truth for this stream yet: add labels.json, or run fingerprint.py, "
                         "or analyze a Twitch VOD with TWITCH_CLIENT_ID/SECRET set.")
    rows, today = [], dt.date.today().isoformat()
    for src, labels in sources.items():
        sig = {k: v for k, v in signals.items() if k != LEAK.get(src)}
        score = hype.combine(sig, weights)
        print(f"\n## {src}: {len(labels)} real moments (signals: {', '.join(sig)})")
        print(f"{'K':>4} {'precision':>10} {'random':>8} {'recall':>8} {'random':>8}")
        for k in a.k:
            p, r = evaluate(score, labels, k, a.tolerance, spacing)
            bp, br = baseline(labels, n, k, a.tolerance)
            print(f"{k:>4} {p:>10.0%} {bp:>8.0%} {r:>8.0%} {br:>8.0%}")
            rows.append([today, a.stream, src, k, round(p, 3), round(bp, 3), round(r, 3), round(br, 3), "all"])
        k = a.k[len(a.k) // 2]
        p_all, _ = evaluate(score, labels, k, a.tolerance, spacing)
        print(f"What each signal adds (precision@{k} without it):")
        for name in sig:
            p, _ = evaluate(hype.combine({x: v for x, v in sig.items() if x != name}, weights), labels, k, a.tolerance, spacing)
            print(f"   {name:<18} {p:>5.0%}  ({(p_all - p) * 100:+.0f} points from it)")
            rows.append([today, a.stream, src, k, round(p, 3), "", "", "", f"without {name}"])
    path = WORK / "benchmark.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "stream", "truth", "k", "precision", "random_precision", "recall", "random_recall", "signals"])
        w.writerows(rows)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
