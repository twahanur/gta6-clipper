import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from profiles import detect_profile, keyword_score


def test_profiles():
    assert detect_profile("PLAYING EVERY GTA UNTIL I BEAT IT: Grand Theft Auto 5", None) == "gta"
    assert detect_profile("Minecraft hardcore day 100", None) == "minecraft"
    assert detect_profile("Just Chatting with the boys", None) is None
    assert detect_profile("Minecraft", "gta") == "gta"
    assert keyword_score("bro what, the cops are here", "gta") >= 2
    assert keyword_score("the cops are here", None) == 0
    assert keyword_score("no no no no no " * 10, None) <= 5


def test_new_game_profiles():
    assert detect_profile("RADIANT GRIND | Valorant ranked", None) == "valorant"
    assert detect_profile("Free Fire Booyah rush", None) == "freefire"
    assert detect_profile("BGMI live custom rooms", None) == "pubg"
    assert detect_profile("Counter-Strike 2 FACEIT", None) == "cs2"
    assert keyword_score("insane clutch, 1v4 ace", "valorant") >= 3


def test_bangla_and_hindi_reactions():
    assert keyword_score("ও ভাই কি হলো এটা", None) >= 1
    assert keyword_score("अरे भाई क्या हुआ", None) >= 2
    assert keyword_score("ki holo bhai", None) >= 1
    # "কি" must not match inside "কিছু" (Bangla vowel signs are not \w)
    assert keyword_score("কিছু হলো না", None) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
