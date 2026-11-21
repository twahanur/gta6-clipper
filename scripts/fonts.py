"""Pick a caption font that can actually draw the words being said.

The default caption fonts (Arial Black, Anton) only cover Latin letters, so Bangla, Hindi, Arabic,
Tamil and other scripts come out as empty boxes. For those, a free Noto Sans Bold font is
downloaded once into ~/.clipper/fonts and used for just those words. Pillow's raqm layout engine
shapes the letters (conjuncts, vowel signs, Arabic joining), so the words read correctly.
"""
import os
import urllib.request

from common import HOME_DIR

FONT_DIR = HOME_DIR / "fonts"
NOTO_URL = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{f}/hinted/ttf/{f}-Bold.ttf"

LATIN = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    str(FONT_DIR / "Anton-Regular.ttf"),  # downloaded by setup.sh, for Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# (first code point, last code point, Noto family, right-to-left)
SCRIPTS = [
    (0x0980, 0x09FF, "NotoSansBengali", False),
    (0x0900, 0x097F, "NotoSansDevanagari", False),
    (0x0A00, 0x0A7F, "NotoSansGurmukhi", False),
    (0x0A80, 0x0AFF, "NotoSansGujarati", False),
    (0x0B00, 0x0B7F, "NotoSansOriya", False),
    (0x0B80, 0x0BFF, "NotoSansTamil", False),
    (0x0C00, 0x0C7F, "NotoSansTelugu", False),
    (0x0C80, 0x0CFF, "NotoSansKannada", False),
    (0x0D00, 0x0D7F, "NotoSansMalayalam", False),
    (0x0D80, 0x0DFF, "NotoSansSinhala", False),
    (0x0E00, 0x0E7F, "NotoSansThai", False),
    (0x0600, 0x06FF, "NotoSansArabic", True),
    (0x0750, 0x077F, "NotoSansArabic", True),
    (0xFB50, 0xFDFF, "NotoSansArabic", True),
    (0xFE70, 0xFEFF, "NotoSansArabic", True),
    (0x0590, 0x05FF, "NotoSansHebrew", True),
    (0x0370, 0x03FF, "NotoSans", False),  # Greek
    (0x0400, 0x04FF, "NotoSans", False),  # Cyrillic
    (0x00C0, 0x024F, "NotoSans", False),  # accented Latin (Anton covers most, Noto covers all)
]

_cache = {}
_failed = set()


def script_of(text):
    """(Noto family, right_to_left) for the first non-basic-Latin letter in text, or (None, False)."""
    for ch in text or "":
        cp = ord(ch)
        if cp < 0x00C0:
            continue
        for lo, hi, fam, rtl in SCRIPTS:
            if lo <= cp <= hi:
                return fam, rtl
    return None, False


def font_file(family):
    """Path to a Noto family's bold font, downloaded on first use. None if it can't be fetched."""
    path = FONT_DIR / f"{family}-Bold.ttf"
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    if family in _failed:
        return None
    try:
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(NOTO_URL.format(f=family), tmp)
        tmp.replace(path)
        print(f"   (downloaded the {family} caption font)")
        return str(path)
    except Exception as e:  # captions must never stop a render
        _failed.add(family)
        print(f"   ({family} font download failed, those words may show as boxes: {str(e)[:80]})")
        return None


def latin_file():
    return next((f for f in LATIN if os.path.exists(f)), None)


def font_for(text, size):
    """A PIL font that can draw this word at this size, and whether the word reads right to left."""
    from PIL import ImageFont
    fam, rtl = script_of(text)
    path = (font_file(fam) if fam else None) or latin_file()
    key = (path, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    return _cache[key], rtl
