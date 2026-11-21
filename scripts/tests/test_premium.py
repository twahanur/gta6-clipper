import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import hype
import visual
from report import write_report


def test_comment_timestamps_and_weights():
    ts = hype.comment_timestamps("2:31:05 when he crashed 💀 / 45:10 lol / 99:99 / 30:00:00", 40000)
    assert [t for t, _ in ts] == [9065, 2710]
    assert ts[0][1] == "2:31:05 when he crashed 💀" and ts[1][1].startswith("45:10 lol")
    assert hype.comment_timestamps("Wow 11:07:03", 50000) == [(40023, "Wow 11:07:03")]
    comments = [{"text": "3:00 LMAOOO", "like_count": 900}, {"text": "1:00 2:00 3:00 4:00 5:00", "like_count": 0},
                {"text": "no timestamp here", "like_count": 5000}]
    sig, marks = hype.comment_signal(comments, 600)
    assert sig[180] > 3 * sig[60]  # one liked comment on a moment beats a list of stamps
    assert len(marks) == 1  # the bare list of times with no words and no likes isn't a bookmark


def test_arc_and_kind():
    score = np.zeros(600)
    score[200:290] = 1.0
    length, a, b = hype.arc_length(score, 240)
    assert 80 <= length <= 110 and a <= 200 and b >= 285
    assert hype.moment_kind({"laugh": 30, "shock": 4}, 0.5, 0, "") == "funny"
    assert hype.moment_kind({"fail": 5, "laugh": 2}, 0.5, 0, "") == "fail or death"
    assert hype.moment_kind({}, 2.0, 0, "") == "streamer reaction"


def test_gray_screen_events():
    sat = np.full(400, 0.4)
    sat[300:305] = 0.05  # a death screen drains the color
    ev = visual.gray_events(sat)
    assert ev[300] == 1 and ev.sum() == 1


def test_frame_stats_colorful_vs_gray():
    color = np.zeros((1, visual.H, visual.W, 3), dtype=np.uint8)
    color[..., 0] = 220
    gray = np.full((1, visual.H, visual.W, 3), 120, dtype=np.uint8)
    _, s_color, _ = visual.frame_stats(color)
    _, s_gray, _ = visual.frame_stats(gray)
    assert s_color[0] > 0.8 and s_gray[0] < 0.05


def test_report_writes():
    n = 3600
    score = np.random.default_rng(1).random(n)
    cands = [{"cand": 1, "moment": 1200, "strength": 100, "kind": "funny", "arc": 60, "why": ["chat 30 laugh"],
              "chat_said": ["LMAO <b>"], "bookmarks": [(1195, 12, "20:00 lol")], "suggested": [1190.0, 1220.0],
              "segments": [{"t": 1190.0, "level": "▮▮▮··", "text": "no way", "moment": True}]}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "report.html")
        write_report(p, {"title": "T", "channel": "C", "duration": n, "url": "https://www.youtube.com/watch?v=x"},
                     score, {"voice": score}, cands, 20)
        page = open(p, encoding="utf-8").read()
    assert "t=1200s" in page and "LMAO &lt;b&gt;" in page and 'id="c1"' in page


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
