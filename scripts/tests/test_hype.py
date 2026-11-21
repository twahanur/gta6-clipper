import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import hype


def test_chat_noise_is_not_a_reaction():
    # the messages that made the old chat-speed signal pick random moments
    for m in ("i was here", "Hi", "hiii kai", "hello", "lag", "laggy af", "!time", "A", "b", "67", "yo", "yoo kai",
              "https://shop.example.com"):
        assert hype.classify_chat(m) is None, m


def test_chat_reactions_by_kind():
    assert hype.classify_chat("LMAOOOO") == "laugh"
    assert hype.classify_chat("hahahaha") == "laugh"
    assert hype.classify_chat("😂😂😂") == "laugh"
    assert hype.classify_chat(":face_with_tears_of_joy::face_with_tears_of_joy:") == "laugh"
    assert hype.classify_chat("WTF") == "shock"
    assert hype.classify_chat("no wayyy") == "shock"
    assert hype.classify_chat("??????") == "shock"
    assert hype.classify_chat("F") == "fail"
    assert hype.classify_chat("WASTED") == "fail"
    assert hype.classify_chat("W") == "hype"
    assert hype.classify_chat("clip it") == "hype"
    assert hype.classify_chat("now how are you") is None  # used to match "w "
    assert hype.classify_chat("my pc is broken") is None  # used to match "bro"
    assert hype.classify_chat("solar eclipse") is None    # used to match "clip"


def test_emoji_runs_become_text():
    runs = [{"text": "bro "}, {"emoji": {"emojiId": "💀", "shortcuts": [":skull:"], "isCustomEmoji": False}},
            {"emoji": {"emojiId": "UC/x", "shortcuts": [":face-purple-crying:"], "isCustomEmoji": True}}]
    text = hype.message_text(runs)
    assert text == "bro 💀:face-purple-crying:"
    assert hype.classify_chat(text) == "laugh"


def test_read_chat_and_lag():
    n = 3600
    rng = np.random.default_rng(0)
    moments = list(range(300, 3300, 300))
    lines = []
    for t in range(0, n, 2):  # background: greetings and plain talk
        lines.append((t, "hi" if t % 4 else "what game is this"))
    for m in moments:        # the streamer yells at m; chat laughs 20 seconds later
        for k in range(25):
            lines.append((m + 20 + rng.uniform(0, 6), "LMAOOO"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "chat.json")
        with open(path, "w") as fh:
            for t, text in lines:
                fh.write(json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(int(t * 1000)), "actions": [
                    {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"message": {"runs": [{"text": text}]}}}}}]}}) + "\n")
        counts, total, msgs = hype.read_chat(path, n)
    assert counts["laugh"].sum() == 25 * len(moments)
    assert all(m[1] != "hi" for m in msgs)  # noise isn't kept
    burst = hype.chat_burst(counts)
    voice = np.zeros(n)
    for m in moments:
        voice[m - 2:m + 3] = 3.0
    lag = hype.estimate_lag(burst, voice)
    assert 17 <= lag <= 25, lag
    score = hype.combine({"voice": hype.scale(voice), "chat_reactions": hype.scale(hype.shift_back(burst, lag))})
    peaks = hype.pick_peaks(score, len(moments), 90)
    assert all(min(abs(p - m) for m in moments) <= 6 for p in peaks), peaks


def test_scale_caps_and_floors():
    x = np.zeros(1000)
    x[5] = 100.0
    x[100:110] = 1.0
    s = hype.scale(x)
    assert s.max() <= 3.0
    probs = np.full(1000, 0.02)
    assert hype.scale(probs, floor=0.3).max() < 0.1  # a "laugh" at 2% is noise


def test_moment_end():
    score = np.zeros(200)
    score[100:115] = 1.0
    assert 113 <= hype.moment_end(score, 100) <= 120


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
