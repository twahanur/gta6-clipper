"""Hear WHAT is happening in a stream, not just how loud it is.

Runs Google's free YAMNet sound classifier (AudioSet, 521 sound types) on the user's own
computer through onnxruntime (already installed with faster-whisper). Every 10-second bin gets:
  action   - gunfire, explosions, crashes, skids, sirens, breaking glass, helicopters
  reaction - screaming, shouting, yelling, laughter, cheering
  calm     - plain talking with no reaction in it (cutscene dialogue, reading chat)
High-voltage moments are loud in action and reaction; cutscenes are high in calm.
"""
import subprocess
import urllib.request

import numpy as np

from common import FFMPEG, HOME_DIR

MODEL_DIR = HOME_DIR / "models"
MODEL = MODEL_DIR / "yamnet.onnx"
CLASS_MAP = MODEL_DIR / "yamnet_class_map.csv"
MODEL_URL = "https://huggingface.co/andrelgomes/yamnet-onnx/resolve/main/yamnet.onnx"
CLASS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

SR = 16000
HOP = 0.48  # seconds between YAMNet frames
CHUNK = 600  # seconds of audio per inference call, so an 11-hour stream never sits in memory

ACTION = {
    420: "explosion", 421: "gunfire", 422: "machine gun", 423: "gunfire", 424: "artillery", 430: "boom",
    460: "bang", 463: "crash", 464: "breaking", 435: "glass", 306: "skid", 307: "tire squeal",
    317: "police siren", 318: "siren", 319: "siren", 390: "siren", 304: "car alarm", 333: "helicopter",
}
REACTION = {
    6: "shout", 7: "bellow", 8: "whoop", 9: "yell", 11: "scream", 13: "laughter", 17: "belly laugh", 61: "cheering",
}
SPEECH = 0

_A = np.array(sorted(ACTION))
_R = np.array(sorted(REACTION))


def ensure_model():
    """Download the model once (~16 MB). Returns its path, or None if it can't be fetched."""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for path, url in ((MODEL, MODEL_URL), (CLASS_MAP, CLASS_URL)):
            if not path.exists() or path.stat().st_size == 0:
                tmp = path.with_suffix(path.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(path)
        return MODEL
    except Exception as e:  # a missing model must never stop an analysis
        print(f"   (sound model download failed, skipping sound events: {str(e)[:80]})")
        return None


def frame_groups(probs):
    """[frames, 521] class scores -> [frames, 3] (action, reaction, calm)."""
    action = probs[:, _A].max(axis=1)
    reaction = probs[:, _R].max(axis=1)
    calm = probs[:, SPEECH] * (1 - reaction)
    return np.stack([action, reaction, calm], axis=1)


def _to_bins(groups, bin_idx, nbins):
    out = np.zeros((nbins, 3), dtype=np.float32)
    for b in np.unique(bin_idx):
        if 0 <= b < nbins:
            out[b] = np.percentile(groups[bin_idx == b], 90, axis=0)
    return {"action": out[:, 0], "reaction": out[:, 1], "calm": out[:, 2]}


def bin_scores(probs, frames_per_bin, nbins):
    """Per-bin action/reaction/calm from frame-level class scores (90th percentile inside each bin)."""
    idx = (np.arange(len(probs)) / frames_per_bin).astype(int)
    return _to_bins(frame_groups(probs), idx, nbins)


def combine(groups):
    # talking only counts against a bin when nothing is happening: streamers talk over the action too
    a = groups["action"]
    return np.clip(a + 1.2 * groups["reaction"] - 0.3 * groups["calm"] * (1 - a), 0, None)


def event_signal(audio, nbins, bin_sec=10):
    """Returns (combined score per bin, {action, reaction, calm, labels}) or None without the model."""
    if not ensure_model():
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"   (sound model failed to load, skipping sound events: {str(e)[:80]})")
        return None
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE)
    all_groups, all_idx = [], []
    best = np.full(nbins, -1, dtype=int)  # strongest action/reaction class per bin, for labels
    best_p = np.zeros(nbins, dtype=np.float32)
    t0 = 0.0
    while True:
        buf = p.stdout.read(SR * CHUNK * 4)
        if len(buf) < SR * 4:  # less than a second left
            break
        x = np.frombuffer(buf[: len(buf) // 4 * 4], dtype=np.float32)
        probs = sess.run(["output_0"], {"waveform": x})[0]
        idx = ((t0 + np.arange(len(probs)) * HOP + HOP) // bin_sec).astype(int)
        all_groups.append(frame_groups(probs))
        all_idx.append(idx)
        cols = np.concatenate([_A, _R])
        sub = probs[:, cols]
        top = sub.argmax(axis=1)
        topv = sub.max(axis=1)
        for b, c, v in zip(idx, top, topv):
            if 0 <= b < nbins and v > best_p[b]:
                best_p[b], best[b] = v, cols[c]
        t0 += len(x) / SR
    p.wait()
    if not all_groups:
        return None
    groups = _to_bins(np.concatenate(all_groups), np.concatenate(all_idx), nbins)
    names = {**ACTION, **REACTION}
    groups["labels"] = [f"{names[c]} {v:.2f}" if c >= 0 and v >= 0.15 else "" for c, v in zip(best, best_p)]
    return combine(groups), groups


def top_labels(groups, b):
    """Short human line for one bin, e.g. 'gunfire 0.82 | action 0.82 reaction 0.10 calm 0.05'."""
    if not groups or b >= len(groups["action"]):
        return ""
    lab = groups["labels"][b] if groups.get("labels") else ""
    return (f"{lab + ' | ' if lab else ''}action {groups['action'][b]:.2f} "
            f"reaction {groups['reaction'][b]:.2f} talk {groups['calm'][b]:.2f}")
