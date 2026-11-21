"""One page per analyzed stream: the hype timeline with every signal, and every candidate moment.

analyze.py writes it as report.html next to for_claude.md. Click a dot on the timeline to jump to that
moment's card; every time links to that second of the stream.
"""
import html
import json

import numpy as np

from common import hms

COLORS = {"score": "#ff5c38", "chat_reactions": "#4cc9f0", "voice": "#ffd60a", "laugh_scream": "#f72585",
          "game_action": "#8d99ae", "visual": "#80ed99", "viewer_bookmarks": "#c77dff", "most_replayed": "#ff9f1c",
          "community_clips": "#06d6a0", "already_clipped": "#ef476f", "facecam_motion": "#ffb4a2", "words": "#bde0fe",
          "vocal_reaction": "#ff70a6"}
NAMES = {"score": "hype score", "chat_reactions": "chat reactions", "voice": "streamer's voice",
         "laugh_scream": "laughs and screams", "game_action": "game sounds", "visual": "on screen",
         "viewer_bookmarks": "viewer bookmarks", "most_replayed": "most replayed",
         "community_clips": "Twitch clips", "already_clipped": "already clipped", "facecam_motion": "streamer moving",
         "words": "reaction words", "vocal_reaction": "voice (separated)"}


def esc(s):
    return html.escape(str(s if s is not None else ""))


def time_link(url, t):
    t = int(t)
    if "youtube.com" in url or "youtu.be" in url:
        return url + ("&" if "?" in url else "?") + f"t={t}s"
    if "twitch.tv" in url:
        return url + ("&" if "?" in url else "?") + f"t={t // 3600}h{t % 3600 // 60}m{t % 60}s"
    return url


def lanes_svg(score, signals, cands, width=1200, lane_h=34):
    n = len(score)
    cols = min(width, n)
    per = int(np.ceil(n / cols))

    def series(x):
        v = np.array([x[i:i + per].max() for i in range(0, n, per)])
        top = float(v.max()) or 1.0
        return v / top

    rows = [("score", score)] + [(k, v) for k, v in signals.items()]
    h = lane_h * len(rows) + 46
    parts = [f'<svg viewBox="0 0 {width} {h}" preserveAspectRatio="none" class="lanes" role="img" '
             f'aria-label="Hype timeline">']
    for r, (key, x) in enumerate(rows):
        y0 = 30 + r * lane_h
        v = series(x)
        hh = lane_h - 8 if key != "score" else lane_h - 4
        pts = " ".join(f"{i / max(1, len(v) - 1) * width:.1f},{y0 + hh - val * hh:.1f}" for i, val in enumerate(v))
        parts.append(f'<polyline points="0,{y0 + hh} {pts} {width},{y0 + hh}" fill="{COLORS.get(key, "#888")}" '
                     f'fill-opacity="{0.55 if key == "score" else 0.35}" stroke="{COLORS.get(key, "#888")}" stroke-width="1"/>')
        parts.append(f'<text x="6" y="{y0 + 12}" class="lane">{esc(NAMES.get(key, key))}</text>')
    for hr in range(0, int(n // 3600) + 1):
        x = hr * 3600 / max(1, n - 1) * width
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="20" y2="{h}" class="grid"/>'
                     f'<text x="{x + 3:.1f}" y="14" class="tick">{hr}h</text>')
    for c in cands:
        x = c["moment"] / max(1, n - 1) * width
        parts.append(f'<a href="#c{c["cand"]}"><circle cx="{x:.1f}" cy="24" r="5" class="dot">'
                     f'<title>#{c["cand"]} {hms(c["moment"])} strength {c["strength"]}</title></circle></a>')
    parts.append("</svg>")
    return "".join(parts)


def write_report(path, meta, score, signals, cands, lag):
    url = meta.get("url", "")
    cards = []
    for c in cands:
        lines = []
        for s in c["segments"]:
            cls = ' class="here"' if s.get("moment") else ""
            lines.append(f'<li{cls}><a href="{esc(time_link(url, s["t"]))}" target="_blank">{hms(s["t"])}</a>'
                         f'<span class="bar">{esc(s["level"])}</span>{esc(s["text"])}</li>')
        said = "".join(f"<span>{esc(x)}</span>" for x in c.get("chat_said", []))
        marks = "".join(f'<li>{esc(m[2])} <em>({m[1]} likes)</em></li>' for m in c.get("bookmarks", []))
        marks += "".join(f'<li>already clipped: <a href="{esc(x["url"])}" target="_blank">{esc(x["title"])}</a> '
                         f'<em>({x["views"]:,} views)</em></li>' for x in c.get("already_clipped", []))
        cards.append(f"""
<article id="c{c['cand']}" class="card">
  <header><span class="num">#{c['cand']}</span>
    <a class="moment" href="{esc(time_link(url, c['moment']))}" target="_blank">{hms(c['moment'])}</a>
    <span class="strength" style="--s:{c['strength']}%">strength {c['strength']}</span>
    <span class="kind">{esc(c.get('kind', ''))}</span>
    {f'<span class="arc">hype lasts {c["arc"]}s</span>' if c.get('arc', 0) >= 45 else ''}</header>
  <p class="why">{esc('; '.join(c.get('why', [])))}</p>
  {f'<div class="said">{said}</div>' if said else ''}
  {f'<ul class="marks">{marks}</ul>' if marks else ''}
  <p class="cut">Suggested cut <a href="{esc(time_link(url, c['suggested'][0]))}" target="_blank">{hms(c['suggested'][0])}</a>
    to {hms(c['suggested'][1])} ({c['suggested'][1] - c['suggested'][0]:.0f}s)</p>
  <ol class="tx">{''.join(lines) or '<li>(no speech)</li>'}</ol>
</article>""")
    legend = "".join(f'<span><i style="background:{COLORS.get(k, "#888")}"></i>{esc(NAMES.get(k, k))}</span>'
                     for k in ["score", *signals])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stream Analysis</title><style>
:root{{--bg:#0e0e12;--card:#17171e;--fg:#f2f2f5;--mut:#9a9aa8;--line:#2a2a35;--acc:#ffd60a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px 16px;max-width:1240px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 18px}}
.lanes{{width:100%;height:auto;background:var(--card);border-radius:14px;display:block}}
.lane{{fill:var(--mut);font-size:11px}} .tick{{fill:var(--mut);font-size:10px}} .grid{{stroke:var(--line);stroke-width:1}}
.dot{{fill:var(--acc);stroke:#000;cursor:pointer}} .dot:hover{{r:8}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;color:var(--mut);font-size:13px;margin:10px 0 24px}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}}
.grid2{{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(360px,1fr))}}
.card{{background:var(--card);border-radius:14px;padding:14px;scroll-margin-top:16px}} .card:target{{outline:2px solid var(--acc)}}
header{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}} .num{{color:var(--mut)}}
.moment{{font-weight:800;font-size:18px;color:var(--acc);text-decoration:none}}
.strength{{background:linear-gradient(90deg,#ff5c38 var(--s),#2a2a35 var(--s));border-radius:8px;padding:1px 8px;font-size:12px}}
.kind,.arc{{background:#24242e;border-radius:8px;padding:1px 8px;font-size:12px;color:var(--mut)}}
.why{{color:var(--mut);margin:8px 0}} .said{{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}}
.said span{{background:#13283a;color:#9ad8f5;border-radius:8px;padding:2px 8px;font-size:13px}}
.marks{{margin:6px 0;padding-left:18px;color:#d9b8ff;font-size:13px}} .marks em{{color:var(--mut)}}
.cut{{margin:8px 0}} .cut a{{color:var(--acc)}}
.tx{{list-style:none;padding:0;margin:8px 0 0;max-height:280px;overflow:auto;font-size:13.5px}}
.tx li{{padding:2px 4px;border-radius:6px}} .tx li.here{{background:#3a2a10}}
.tx a{{color:var(--mut);text-decoration:none;margin-right:6px;font-variant-numeric:tabular-nums}}
.bar{{color:#ff8a65;margin-right:6px;font-family:monospace}}
</style></head><body>
<h1>{esc(meta.get('title'))}</h1>
<p class="sub">{esc(meta.get('channel'))} · {hms(meta.get('duration', 0))} · {len(cands)} candidate moments{f' · chat runs {lag}s behind the stream (already corrected)' if lag else ''}</p>
{lanes_svg(score, signals, cands)}
<div class="legend">{legend}</div>
<div class="grid2">{''.join(cards)}</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


if __name__ == "__main__":  # rebuild a report from a stream folder's candidates.json (no timeline)
    import sys
    d = sys.argv[1]
    doc = json.load(open(f"{d}/candidates.json", encoding="utf-8"))
    n = int(doc["meta"]["duration"]) or 1
    write_report(f"{d}/report.html", doc["meta"], np.zeros(n), {}, doc["candidates"], doc.get("chat_lag", 0))
