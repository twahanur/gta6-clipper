# Sound-event hype + video folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every analysis scores each 10 s bin for action/reaction sounds (YAMNet) plus transcript keywords, and every rendered clip is also copied to `~/Clipper/videos/NNN_<slug>.mp4`.

**Architecture:** New `sound_events.py` (model download + per-bin scores) and `profiles.py` (keyword lists, profile detection). `analyze.py` adds them as signals; `cut.py` publishes finished clips to the serial video folder.

**Tech Stack:** Python 3.12, numpy, onnxruntime (already installed by faster-whisper), YAMNet ONNX (`andrelgomes/yamnet-onnx`, Apache-2.0, input `waveform` f32 16 kHz, output `output_0` [frames, 521], hop 0.48 s).

**Spec:** `docs/2026-11-21-sound-events-design.md`

## Global Constraints
- No new pip packages. Model lives in `~/.clipper/models/yamnet.onnx` + `yamnet_class_map.csv`.
- Missing/broken model never stops analysis: print one line, skip the signal.
- Video folder: `WORK / "videos"`, names `001_<slug>.mp4` (3-digit serial, continues across runs).
- No commits (user hasn't asked).

## Review Focus
- Re-rendering a clip (`--only 03`) must overwrite its existing serial file, not take a new number.
- A stream with long music/BRB stretches must not score as "action".
- 11-hour audio must stream in chunks (never load all 640M samples).
- Non-GTA streams: no GTA words applied unless title matches or `--game gta`.
- Chat `.part` left over: retry once after deleting it.

---

### Task 1: sound_events.py
**Files:** Create `scripts/sound_events.py`, test `scripts/tests/test_sound_events.py`
**Produces:**
- `ensure_model() -> pathlib.Path | None` (downloads both files if missing)
- `ACTION`, `REACTION`, `CALM` class-index tuples (gunshot 421, machine gun 422, fusillade 423, artillery 424, explosion 420, boom 430, bang 460, smash/crash 463, breaking 464, glass 435, skidding 306, tire squeal 307, sirens 317-319/390, car alarm 304, helicopter 333 | shout 6, bellow 7, whoop 8, yell 9, screaming 11, laughter 13, belly laugh 17, cheering 61 | speech 0 minus reaction, music 132)
- `bin_scores(probs: np.ndarray, frames_per_bin: float) -> dict[str, np.ndarray]` keys `action`, `reaction`, `calm` (per bin: 90th-percentile of max-class prob in group)
- `event_signal(audio: Path, nbins: int, bin_sec: int = 10) -> tuple[np.ndarray, dict] | None` → combined score `action + 1.2*reaction - 0.5*calm` (clipped ≥0) and the per-group arrays; streams ffmpeg 16 kHz f32 in 10-minute chunks.
- `top_labels(groups: dict, b: int) -> str` e.g. `"gunfire 0.82, scream 0.41"` for for_claude.md.
- [ ] Test: synthetic probs (gunshot high in bin 2, speech-only in bin 0) → `bin_scores` action[2] > action[0], calm[0] > calm[2].
- [ ] Implement, run `python -I scripts/tests/test_sound_events.py` → `ok`.
- [ ] Smoke: `event_signal` on Kai audio 10:03:00-10:06:00 shows action/reaction > 0.

### Task 2: profiles.py
**Files:** Create `scripts/profiles.py`, test `scripts/tests/test_profiles.py`
**Produces:** `REACTION_WORDS`, `PROFILES = {"gta": [...]}`, `detect_profile(title: str, forced: str | None) -> str | None` (gta if title matches `gta|grand theft auto`), `keyword_score(text: str, profile: str | None) -> float` (count of hits, capped at 5).
- [ ] Test: `detect_profile("PLAYING ... Grand Theft Auto 5", None) == "gta"`, `detect_profile("Minecraft", None) is None`; `keyword_score("bro what the cops", "gta") >= 2`, `keyword_score("cops", None) == 0`.
- [ ] Implement, run test → `ok`.

### Task 3: analyze.py wiring
**Files:** Modify `scripts/analyze.py`
- [ ] `--top` default 40; add `--game`.
- [ ] After loudness: `sound_events` signal = `robust_z(combined)`; weights `loudness .30, sound_events .30, chat_speed .25, most_replayed .15`.
- [ ] `download_chat`: if `chat.live_chat.json.part` exists, delete and retry once.
- [ ] Each candidate gets `keywords` = keyword_score over its transcript and `sounds` = top_labels at peak; header line shows both; candidates in for_claude.md sorted by `hype + 8*keywords`.
- [ ] Run on Kai stream (audio cached) → finishes, for_claude.md shows `sounds:` lines.

### Task 4: serial video folder
**Files:** Modify `scripts/common.py` (`next_serial(folder) -> int`), `scripts/cut.py` (after each successful render).
- [ ] `publish(out: str, run_dir: str, key: str, name: str) -> str`: reads `<run>/videos.json` {key: serial}; reuse serial if present (delete old `NNN_*` file), else `next_serial`; copy to `WORK/videos/NNN_<slug>.mp4`; print path.
- [ ] Test `next_serial` on temp dir with `001_a.mp4`, `007_b.mp4` → 8; empty → 1.
- [ ] Copy existing run `2026-11-21_191618` clips in as 001-003 (run publish for it).

### Task 5: setup.sh + SKILL.md
- [ ] setup.sh step downloads the model (calls `ensure_model()`).
- [ ] SKILL.md: mention sound events, `--game`, and the `~/Clipper/videos` folder.
