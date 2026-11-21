"""Words that mark a high-voltage moment in a transcript.

REACTION_WORDS apply to every stream. A game profile adds that game's own event words; it turns
on by itself when the stream title names the game, or with analyze.py --game <name>.
"""
import re

REACTION_WORDS = (
    "oh my god", "oh my gosh", "what the fuck", "what the hell", "holy shit", "bro what", "bro, what",
    "no no no", "no way", "let's go", "lets go", "are you serious", "yo yo yo", "run run",
    "go go go", "i'm dead", "im dead", "help me", "get out", "oh shit", "wait wait",
)

PROFILES = {
    "gta": {
        "title": r"\bgta\b|grand theft auto",
        "words": (
            "cops", "police", "wanted", "stars", "chase", "chopper", "helicopter", "swat", "military", "tank",
            "wasted", "busted", "mission failed", "heist", "bank", "explode", "explosion", "blew up", "crash",
            "shoot", "shot", "gun", "rpg", "sticky bomb", "jet", "plane", "parachute", "lost them", "evade",
            "dead", "died", "kill",
        ),
    },
}


def detect_profile(title, forced=None):
    if forced:
        return forced.lower()
    for name, prof in PROFILES.items():
        if re.search(prof["title"], title or "", re.I):
            return name
    return None


def _count(text, words):
    return sum(len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text)) for w in words)


def keyword_score(text, profile=None):
    """Number of reaction (and game) word hits in the text, capped at 5."""
    t = (text or "").lower()
    n = _count(t, REACTION_WORDS)
    if profile in PROFILES:
        n += _count(t, PROFILES[profile]["words"])
    return float(min(5, n))
