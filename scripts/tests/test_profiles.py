import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from profiles import detect_profile, keyword_score

def test_profiles():
    assert detect_profile("PLAYING EVERY GTA UNTIL I BEAT IT: Grand Theft Auto 5", None) == "gta"
    assert detect_profile("Minecraft hardcore day 100", None) is None
    assert detect_profile("Minecraft", "gta") == "gta"
    assert keyword_score("bro what, the cops are here", "gta") >= 2
    assert keyword_score("the cops are here", None) == 0
    assert keyword_score("no no no no no " * 10, None) <= 5

if __name__ == "__main__":
    test_profiles(); print("ok")
