# Sound-event hype + video folder (design, approved 2026-11-21)

## Why
Loudness alone mistakes loud GTA cutscenes for hype and can't tell a gunfight, a crash or a scream
from talking. The user wants the clipper to catch real high-voltage moments, on **every stream**.

## Goals
- Every `analyze.py` run adds a `sound_events` signal: a free AudioSet classifier (YAMNet, ONNX,
  run with the onnxruntime that faster-whisper already installs) scores each second for action
  sounds (gunshot, explosion, siren, crash/skid, glass) and reaction sounds (scream, shout,
  laughter). Calm dialogue (cutscenes) lowers the score.
- Transcript keyword bonus when ranking windows: reaction words ("bro what", "oh my god",
  "no no no", "let's go") always; game words (wanted, cops, wasted, busted, heist) when the
  stream title is about GTA (a `gta` profile, auto-on from the title, or `--game gta`).
- Retry the chat replay download when a partial `.part` file is left behind.
- Default `--top` 40 so more windows get read.
- Finished clips are also saved to `~/Clipper/videos/` with a serial number across all runs:
  `001_<hook-slug>.mp4`, `002_...` (the user's request). Existing clips get copied in as 001-003.

## Non-goals / anti-goals
- No video-frame OCR (would need the whole video downloaded).
- No new heavy installs (no torch/tensorflow). If the model can't be downloaded, analysis still
  runs on the other signals and says so.

## Approaches considered
1. AudioSet sound classifier (YAMNet ONNX) - **picked**: real labels for gunshots/screams, no new
   Python packages, ~15 MB model, ~10-15 extra minutes on an 11-hour stream.
2. Transcript keywords only - cheap but blind to gunfire/crashes.
3. On-screen WASTED/BUSTED detection - needs the full video, too heavy.

## Builder's calls
Model source, exact class lists and weights, slug format.
