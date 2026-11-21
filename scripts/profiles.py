"""Words that mark a high-voltage moment in a transcript (chat is read by hype.py).

REACTION_WORDS apply to every stream, in English, Bangla and Hindi (both in their own script, which
is what Whisper writes, and romanized, which people sometimes say too). A game profile adds that game's
own event words; it turns on by itself when the stream title names the game, or with
analyze.py --game <name>.
"""
import re

REACTION_WORDS = (
    # English
    "oh my god", "oh my gosh", "what the fuck", "what the hell", "holy shit", "bro what", "bro, what",
    "no no no", "no way", "let's go", "lets go", "are you serious", "yo yo yo", "run run",
    "go go go", "i'm dead", "im dead", "help me", "get out", "oh shit", "wait wait",
    # Bangla
    "কি হলো", "কী হলো", "কি হইলো", "কি হচ্ছে", "কী হচ্ছে", "কি করলি", "কী করলি", "না না না", "ওরে বাবা",
    "ও মাই গড", "আল্লাহ রে", "মরে গেলাম", "শেষ আমি", "পালা পালা", "বাঁচাও", "সর্বনাশ", "আরে ভাই", "চল চল",
    "ki holo", "ki hoilo", "ki korli", "ore baba", "sorbonash", "are bhai", "more gelam",
    # Hindi
    "क्या हुआ", "क्या हो रहा", "अरे भाई", "नहीं नहीं", "मर गया", "भागो", "बचाओ", "ओ तेरी", "हे भगवान",
    "क्या बात है", "चलो चलो", "kya hua", "arey bhai", "are bhai", "o teri", "bhago", "mar gaya",
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
    "valorant": {
        "title": r"\bvalorant\b|\bvalo\b",
        "words": ("ace", "clutch", "headshot", "one tap", "spike", "defuse", "planted", "ult", "flick",
                  "1v5", "1v4", "1v3", "team ace", "eco", "op", "operator", "whiff", "thrifty", "flawless"),
    },
    "cs2": {
        "title": r"\bcs2\b|\bcs:?go\b|counter[- ]?strike",
        "words": ("ace", "clutch", "headshot", "one tap", "awp", "no scope", "noscope", "defuse", "bomb",
                  "planted", "eco", "deagle", "1v5", "1v4", "1v3", "flick", "wallbang", "ninja defuse"),
    },
    "fortnite": {
        "title": r"\bfortnite\b",
        "words": ("victory royale", "storm", "box", "build", "edit", "crown", "elim", "eliminated",
                  "third party", "shotgun", "sniper", "no scope", "final circle", "zero build", "knocked"),
    },
    "minecraft": {
        "title": r"\bminecraft\b",
        "words": ("creeper", "diamond", "diamonds", "nether", "ender dragon", "lava", "warden", "hardcore",
                  "totem", "died", "dead", "fell", "mobs", "skeleton", "netherite", "wither", "trapped"),
    },
    "freefire": {
        "title": r"free ?fire|\bff max\b",
        "words": ("booyah", "headshot", "airdrop", "gloo wall", "zone", "rush", "knocked", "one tap",
                  "squad wipe", "last zone", "awm", "m1887", "1v4", "1v3", "clutch"),
    },
    "pubg": {
        "title": r"\bpubg\b|\bbgmi\b|battlegrounds",
        "words": ("chicken dinner", "winner winner", "airdrop", "zone", "knocked", "squad wipe", "awm",
                  "kar98", "sniper", "headshot", "red zone", "last circle", "1v4", "1v3", "clutch", "grenade"),
    },
    "cod": {
        "title": r"warzone|call of duty|\bcod\b|black ops|modern warfare",
        "words": ("gulag", "nuke", "loadout", "squad wipe", "killstreak", "uav", "sniper", "headshot",
                  "quickscope", "quick scope", "no scope", "1v4", "1v3", "clutch", "last circle", "plates"),
    },
    "apex": {
        "title": r"\bapex\b",
        "words": ("champion", "third party", "knocked", "squad wipe", "kraber", "wingman", "headshot",
                  "ring", "final ring", "respawn", "banner", "1v3", "clutch", "20 bomb", "4k damage"),
    },
    "football": {
        "title": r"\bfifa\b|\bea ?fc\b|\bfc ?2[4-9]\b|\befootball\b|\bpes\b",
        "words": ("goal", "golazo", "penalty", "red card", "offside", "bicycle kick", "free kick", "last minute",
                  "equalizer", "pack", "walkout", "hat trick", "own goal", "var"),
    },
}

# A "word character" for matching: \w plus the Indic blocks, because Bangla and Hindi vowel signs
# aren't \w and would otherwise let "কি" match inside "কিছু".
_WCH = r"[\wऀ-෿]"


def _pattern(words):
    return re.compile("|".join(r"(?<!" + _WCH + r")" + re.escape(w) + r"(?!" + _WCH + r")" for w in words))


_REACTION_RE = _pattern(REACTION_WORDS)
_PROFILE_RE = {name: _pattern(p["words"]) for name, p in PROFILES.items()}


def detect_profile(title, forced=None):
    if forced:
        return forced.lower()
    for name, prof in PROFILES.items():
        if re.search(prof["title"], title or "", re.I):
            return name
    return None


_REPEAT_RE = re.compile(r"(?<!\w)(\w{2,})(?:[ ,!.]+\1(?!\w)){2,}")  # "no no no", "go, go, go", "bro bro bro"


def keyword_score(text, profile=None):
    """Number of reaction (and game) word hits in the text, plus repeated-word bursts, capped at 5."""
    t = (text or "").lower()
    n = len(_REACTION_RE.findall(t)) + len(_REPEAT_RE.findall(t))
    if profile in _PROFILE_RE:
        n += len(_PROFILE_RE[profile].findall(t))
    return float(min(5, n))

