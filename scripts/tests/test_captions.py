import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cut
from common import clean_word
from fonts import script_of


def test_clean_word_keeps_vowel_signs():
    assert clean_word(" কিছু") == "কিছু"           # Bangla vowel signs used to be stripped to "কছ"
    assert clean_word("हिन्दी,") == "हिन्दी,"
    assert clean_word("“Bro!”") == "Bro!"
    assert clean_word("না।") == "না।"


def test_script_detection():
    assert script_of("HELLO") == (None, False)
    assert script_of("ভাই") == ("NotoSansBengali", False)
    assert script_of("भाई") == ("NotoSansDevanagari", False)
    assert script_of("مرحبا") == ("NotoSansArabic", True)


def test_caption_chunks_keep_word_timings():
    words = [(0.0, 0.3, "no"), (0.3, 0.6, "way"), (0.6, 0.9, "bro."), (2.0, 2.4, "what")]
    ch = cut.caption_chunks(words)
    assert [c[2] for c in ch] == ["NO WAY BRO", "WHAT"]
    assert ch[0][3] == [(0.0, 0.3, "NO"), (0.3, 0.6, "WAY"), (0.6, 0.9, "BRO")]


def test_karaoke_track_lights_each_word():
    chunks = cut.caption_chunks([(4.0, 4.3, "no"), (4.3, 4.6, "way"), (4.6, 5.0, "bro")])
    with tempfile.TemporaryDirectory() as t:
        lst = cut.build_overlay_track(t, "HOOK", chunks, 6.0, 1250, karaoke=True)
        karaoke_frames = len([n for n in os.listdir(t) if n.endswith(".png")])
        assert os.path.exists(lst)
    with tempfile.TemporaryDirectory() as t:
        cut.build_overlay_track(t, "HOOK", chunks, 6.0, 1250, karaoke=False)
        plain_frames = len([n for n in os.listdir(t) if n.endswith(".png")])
    # hook, empty, then one frame per lit word (karaoke) or one for the whole caption (plain)
    assert karaoke_frames == plain_frames + 2


def test_focus_x_zero_means_far_left():
    vf, _ = cut.layout_filter(1920, 1080, {"facecam": None, "focus_x": 0})
    assert "crop=606:1080:0:0" in vf
    vf, _ = cut.layout_filter(1920, 1080, {"facecam": None})
    assert ":657:0" in vf  # no focus_x: centered


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
