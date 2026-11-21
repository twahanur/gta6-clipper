import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import hype


def test_twitch_community_clips_with_mock_api(monkeypatch=None):
    import twitch
    os.environ["TWITCH_CLIENT_ID"], os.environ["TWITCH_CLIENT_SECRET"] = "id", "secret"
    calls = []

    def http(url, data=None, headers=None):
        calls.append(url)
        if "oauth2/token" in url:
            return {"access_token": "tok"}
        if "/videos?" in url:
            return {"data": [{"user_id": "42", "created_at": "2026-11-20T18:00:00Z", "duration": "3h2m5s"}]}
        if "after=c2" in url:
            return {"data": [{"video_id": "999", "vod_offset": 7200, "duration": 30, "view_count": 50, "title": "b"}]}
        return {"data": [{"video_id": "999", "vod_offset": 3600, "duration": 25, "view_count": 900, "title": "a"},
                         {"video_id": "111", "vod_offset": 10, "duration": 30, "view_count": 5, "title": "other vod"},
                         {"video_id": "999", "vod_offset": None, "duration": 30, "view_count": 5, "title": "no offset"}],
                "pagination": {"cursor": "c2"}}
    try:
        clips = twitch.community_clips("https://www.twitch.tv/videos/999", http=http)
    finally:
        del os.environ["TWITCH_CLIENT_ID"], os.environ["TWITCH_CLIENT_SECRET"]
    assert [c["offset"] for c in clips] == [3600, 7200]
    assert twitch.parse_duration("11h10m20s") == 40220
    assert twitch.community_clips("https://www.twitch.tv/videos/999", http=http) is None  # no credentials


def test_fingerprint_finds_a_quieter_noisier_copy():
    from fingerprint import Matcher, SR, prints_from_pcm
    rng = np.random.default_rng(0)
    t = np.arange(SR * 600) / SR
    x = (rng.standard_normal(len(t)) * np.repeat(rng.random(600 * 5) + 0.1, SR // 5)).astype(np.float32)
    x += (0.5 * np.sin(2 * np.pi * np.repeat(rng.uniform(200, 2000, 600 * 2), SR // 2) * t)).astype(np.float32)
    clip = x[SR * 400:SR * 430] * 0.3 + 0.2 * rng.standard_normal(SR * 30).astype(np.float32)
    at, conf = Matcher(prints_from_pcm(x)).locate(prints_from_pcm(clip))
    assert abs(at - 400) < 0.5 and conf > 15


def test_learn_weights_rewards_signals_that_go_with_views():
    from learn import learn_weights
    rng = np.random.default_rng(1)
    matched = []
    for i in range(20):
        chat = rng.random()
        matched.append({"views": int(1000 * np.exp(3 * chat)), "signals": json.dumps({"chat_reactions": chat,
                        "game_action": rng.random(), "strength": 50})})
    w, corr = learn_weights(matched)
    assert w["chat_reactions"] > 1.3 and 0.6 <= w["game_action"] <= 1.3
    assert learn_weights(matched[:5]) == (None, None)


def test_weights_change_the_score():
    sig = {"voice": np.array([0, 1.0, 0]), "chat_reactions": np.array([1.0, 0, 0])}
    base = hype.combine(sig)
    more_chat = hype.combine(sig, {"chat_reactions": 2.0})
    assert more_chat[0] > base[0] and more_chat[1] == base[1]


def test_words_signal():
    segs = [{"start": 100.0, "end": 103.0, "text": "no no no oh my god"}, {"start": 200.0, "end": 202.0, "text": "okay so"}]
    s = hype.words_signal(segs, 300)
    assert s[101] > 0 and s[201] == 0


def test_suggest_cut_on_sentence_boundaries():
    from analyze import suggest_cut
    score = np.zeros(400)
    score[200:212] = 1.0
    segs = [{"start": 185.0, "end": 189.0, "text": "So we go in."}, {"start": 190.5, "end": 195.0, "text": "Wait what is that"},
            {"start": 195.2, "end": 205.0, "text": "NO NO NO he got me!"}, {"start": 205.1, "end": 214.0, "text": "bro I can't"},
            {"start": 215.5, "end": 220.0, "text": "anyway"}]
    a, b = suggest_cut(score, 200, segs, 400)
    assert 190 <= a <= 191        # starts on the line after the pause, about 8 s before the peak
    assert 213 <= b <= 215        # ends after the reaction, on a sentence end


def test_cluster_speakers():
    from deep import cluster_speakers
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(192), rng.standard_normal(192)
    prints = []
    for i in range(40):
        base = a if (i // 5) % 2 == 0 else b
        e = base + 0.3 * rng.standard_normal(192)
        prints.append((i * 0.75, e / np.linalg.norm(e)))
    labels = cluster_speakers(prints)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] != labels[5]


def test_facecam_helpers():
    from facecam import _best_line, iou, stable_face
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) > 0.99
    per = [[(100, 100, 40, 40, 0.9), (500, 300, 30, 30, 0.8)], [(102, 101, 40, 40, 0.9)], [(101, 99, 41, 40, 0.9), (700, 50, 30, 30, 0.8)]]
    box, n = stable_face(per)
    assert n == 3 and abs(box[0] - 101) <= 1
    prof = np.ones(300)
    prof[150], prof[220] = 20.0, 15.0  # furniture line inside, border line farther out
    assert _best_line(prof, 120, 260, 299, +1)[0] == 220
    assert _best_line(np.ones(300), 0, 100, 0, -1)[0] == 0  # no line: the picture's edge


def test_live_chat_offsets_outside_replay():
    line = {"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
        "message": {"runs": [{"text": "LMAOOO"}]}}}}}]}, "videoOffsetTimeMsec": "65000", "isLive": True}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.json")
        with open(p, "w") as fh:
            fh.write(json.dumps(line) + "\n")
        counts, total, msgs = hype.read_chat(p, 120)
    assert counts["laugh"][65] == 1 and msgs[0][0] == 65.0


def test_clip_signal_caps_mega_views():
    sig = hype.clip_signal([{"offset": 10, "duration": 5, "views": 10 ** 9}, {"offset": 50, "duration": 5, "views": 10 ** 5}], 100)
    assert sig[12] <= 13.0 and sig[52] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
