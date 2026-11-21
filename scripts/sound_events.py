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
    6: "shout", 7: "bellow", 8: "whoop", 9: "yell", 11: "scream", 13: "laughter", 15: "giggle", 16: "snicker",
    17: "belly laugh", 18: "chuckle", 61: "cheering",
}
SPEECH = 0
MUSIC = 132

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


def stream_features(audio, with_model=True):
    """One pass over the whole audio, one value per second:
      loud     overall level (dB)
      voice    level in the voice band, 300 to 3400 Hz (dB): the streamer yelling, not the game's bass
      bright   spectral brightness (Hz): yelling and laughing are brighter than talking
    and with the YAMNet model:
      speech, reaction (laugh/shout/scream), action (gunfire, explosion, crash...), music (copyright risk)
      and a label per second.
    """
    sess = None
    if with_model and ensure_model():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"   (sound model failed to load, using loudness and voice only: {str(e)[:80]})")
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    band = (freqs >= 300) & (freqs <= 3400)
    win = np.hanning(SR).astype(np.float32)
    out = {k: [] for k in ("loud", "voice", "bright", "speech", "reaction", "action", "music", "label_c", "label_p")}
    cols = np.concatenate([_A, _R])
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE)
    while True:
        buf = p.stdout.read(SR * CHUNK * 4)
        n = len(buf) // 4 // SR  # whole seconds in this chunk
        if n < 1:
            break
        x = np.frombuffer(buf[: n * SR * 4], dtype=np.float32)
        fr = x.reshape(n, SR)
        out["loud"].append(10 * np.log10(np.mean(fr * fr, axis=1) + 1e-10))
        spec = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
        out["voice"].append(10 * np.log10(spec[:, band].sum(axis=1) + 1e-10))
        out["bright"].append((spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-10))
        del spec
        if sess is not None:
            probs = sess.run(["output_0"], {"waveform": x})[0]
            sec = np.minimum((np.arange(len(probs)) * HOP + HOP).astype(int), n - 1)
            for key, val in (("speech", probs[:, SPEECH]), ("reaction", probs[:, _R].max(axis=1)),
                             ("action", probs[:, _A].max(axis=1)), ("music", probs[:, MUSIC])):
                a = np.zeros(n, dtype=np.float32)
                np.maximum.at(a, sec, val)
                out[key].append(a)
            sub = probs[:, cols]
            best_c = np.full(n, -1)
            best_p = np.zeros(n, dtype=np.float32)
            for s_, c, v in zip(sec, sub.argmax(axis=1), sub.max(axis=1)):
                if v > best_p[s_]:
                    best_p[s_], best_c[s_] = v, cols[c]
            out["label_c"].append(best_c)
            out["label_p"].append(best_p)
    p.wait()
    feat = {k: np.concatenate(v) for k, v in out.items() if v}
    if "label_c" in feat:
        names = {**ACTION, **REACTION}
        feat["labels"] = [names[c] if c >= 0 and v >= 0.3 else "" for c, v in zip(feat.pop("label_c"), feat.pop("label_p"))]
    return feat
