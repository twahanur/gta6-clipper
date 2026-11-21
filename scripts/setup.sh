#!/bin/bash
# Clipper setup.
# Installs everything into ~/.clipper so nothing else on your computer changes.
# Safe to run again: it skips anything that's already installed.
set -e
command -v curl >/dev/null 2>&1 || { echo "curl is required. Install it and run this again."; exit 1; }
HOME_DIR="$HOME/.clipper"
mkdir -p "$HOME_DIR/bin"
export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH"

echo "==> 1/5 Python manager (uv)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "    ok"

echo "==> 2/5 Python 3.12 environment"
if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
  uv venv --python 3.12 "$HOME_DIR/venv" >/dev/null
fi
echo "    ok"

echo "==> 3/5 Tools: yt-dlp, Whisper, ffmpeg, Pillow"
uv pip install --python "$HOME_DIR/venv/bin/python" -q -U \
  "yt-dlp[default]" faster-whisper imageio-ffmpeg pillow numpy
FF=$("$HOME_DIR/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" "$HOME_DIR/bin/ffmpeg"
echo "    ok"

echo "==> 4/5 JavaScript runtime for YouTube downloads (deno)"
if ! command -v deno >/dev/null 2>&1 && [ ! -x "$HOME_DIR/venv/bin/deno" ]; then
  uv pip install --python "$HOME_DIR/venv/bin/python" -q deno >/dev/null 2>&1 \
    || curl -fsSL https://deno.land/install.sh | sh -s -- -y --no-modify-path >/dev/null 2>&1 \
    || echo "    deno install failed; YouTube downloads may fail. Install deno from deno.com and run setup again."
fi
echo "    ok"

echo "==> 5/5 Whisper speech models (one time download)"
"$HOME_DIR/venv/bin/python" - <<'PY'
from faster_whisper import WhisperModel
for m in ("base", "small"):
    WhisperModel(m, device="cpu", compute_type="int8")
print("    ok")
PY

echo "==> Sound classifier (YAMNet, hears gunfire, explosions, screams, laughter; one time ~16 MB)"
(cd "$(dirname "$0")" && "$HOME_DIR/venv/bin/python" -c "from sound_events import ensure_model; print('    ok' if ensure_model() else '    skipped: analysis will use loudness only')")

echo "==> Caption font (Anton, free open-source font, used when Arial Black isn't installed)"
mkdir -p "$HOME_DIR/fonts"
if [ ! -s "$HOME_DIR/fonts/Anton-Regular.ttf" ]; then
  curl -fsSL -o "$HOME_DIR/fonts/Anton-Regular.ttf" \
    "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf" \
    || echo "    font download failed; captions will use a system font"
fi
echo "    ok"

echo "==> Self-test"
"$HOME_DIR/venv/bin/python" -c "import yt_dlp, faster_whisper, PIL, numpy; print('    python packages ok, yt-dlp', yt_dlp.version.__version__)"
"$HOME_DIR/bin/ffmpeg" -hide_banner -version | head -1 | sed 's/^/    /'
if command -v deno >/dev/null 2>&1 || [ -x "$HOME_DIR/venv/bin/deno" ] || [ -x "$HOME/.deno/bin/deno" ]; then echo "    deno ok"; else echo "    WARNING: deno missing, YouTube downloads may fail"; fi
if "$HOME_DIR/bin/ffmpeg" -hide_banner -encoders 2>/dev/null | grep -q libx264; then echo "    h264 encoder ok"; else echo "    libx264 missing, will use the built-in encoder"; fi
echo "READY: Clipper is set up."
