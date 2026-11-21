"""analyze.py --deep: slower, heavier listening on the candidate windows only.

  * Vocal separation (Demucs): pulls the voices out of the game audio and music, then listens for
    laughing, yelling and screaming in the voices alone. On a GTA stream the sound classifier heard
    the streamer's reaction far better this way (laugh/shout 0.79 vs 0.60 in the mix).
  * Speaker diarization (SpeechBrain ECAPA voice prints): who is talking when, so podcast and
    Just Chatting transcripts show [A] / [B], and fast back-and-forth (arguments, banter) is found.

Needs the deep extras: bash scripts/setup.sh --deep (PyTorch CPU, Demucs, SpeechBrain, ~1.5 GB).
About 20 seconds per candidate on a CPU; much faster with an NVIDIA GPU.
"""
import subprocess

import numpy as np

from common import FFMPEG, HOME_DIR

SR = 16000


def available():
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("torch", "torchaudio", "demucs", "speechbrain"))


def _pcm(audio, start, dur, sr, channels):
    raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(audio),
                          "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T.copy()


class Deep:
    def __init__(self):
        import torch
        from demucs.pretrained import get_model
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sep = get_model("htdemucs").to(self.device).eval()
        self._spk = None

    def vocals(self, audio, start, dur):
        """The voices alone, 16 kHz mono."""
        import torchaudio
        from demucs.apply import apply_model
        x = self.torch.from_numpy(_pcm(audio, start, dur, self.sep.samplerate, 2))
        with self.torch.no_grad():
            out = apply_model(self.sep, x[None], device=self.device, shifts=0, split=True, overlap=0.1, progress=False)[0]
        voc = out[self.sep.sources.index("vocals")].mean(0).cpu()
        return torchaudio.functional.resample(voc, self.sep.samplerate, SR).numpy().astype(np.float32)

    def speaker_model(self):
        if self._spk is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                       savedir=str(HOME_DIR / "models" / "ecapa"),
                                                       run_opts={"device": self.device})
        return self._spk

    def voice_prints(self, x, win=1.5, hop=0.75):
        """[(t, embedding)] every `hop` seconds over voiced parts of a 16 kHz signal."""
        spk = self.speaker_model()
        out = []
        n, w, h = len(x), int(win * SR), int(hop * SR)
        chunks, times = [], []
        for s in range(0, max(1, n - w), h):
            seg = x[s:s + w]
            if np.sqrt(np.mean(seg ** 2)) < 0.01:  # silence: no voice to print
                continue
            chunks.append(seg)
            times.append(s / SR + win / 2)
        for i in range(0, len(chunks), 32):
            batch = self.torch.from_numpy(np.stack(chunks[i:i + 32]))
            with self.torch.no_grad():
                emb = spk.encode_batch(batch.to(self.device)).squeeze(1).cpu().numpy()
            out += list(zip(times[i:i + 32], emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)))
        return out


def cluster_speakers(prints, threshold=0.55):
    """Greedy cosine clustering of voice prints into speakers. Returns labels per print."""
    cents, counts, labels = [], [], []
    for _, e in prints:
        sims = [float(e @ (c / np.linalg.norm(c))) for c in cents]
        k = int(np.argmax(sims)) if sims else -1
        if k >= 0 and sims[k] >= threshold:
            cents[k] = cents[k] + e
            counts[k] += 1
        else:
            cents.append(e.copy())
            counts.append(1)
            k = len(cents) - 1
        labels.append(k)
    # speakers heard only a couple of times are noise (a game voice, a clip): fold them into the nearest
    keep = [i for i, c in enumerate(counts) if c >= max(3, 0.05 * len(prints))]
    if not keep:
        return [0] * len(labels)
    remap = {}
    for i in range(len(cents)):
        if i in keep:
            remap[i] = keep.index(i)
        else:
            sims = [float(cents[i] @ cents[j]) / (np.linalg.norm(cents[i]) * np.linalg.norm(cents[j])) for j in keep]
            remap[i] = int(np.argmax(sims))
    return [remap[k] for k in labels]


def reaction_in_vocals(voc, moment_offset, sess=None):
    """How hard the voices laugh/yell near the moment: YAMNet on the separated vocals, plus how much
    louder the voice gets than its own median in the window. 0 to ~2."""
    from sound_events import MODEL, _R, HOP, ensure_model
    import onnxruntime as ort
    if sess is None:
        ensure_model()
        sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    probs = sess.run(["output_0"], {"waveform": voc})[0]
    t = np.arange(len(probs)) * HOP + HOP
    near = np.abs(t - moment_offset) <= 8
    react = float(probs[near][:, _R].max()) if near.any() else 0.0
    sec = len(voc) // SR
    rms = np.sqrt(np.mean(voc[: sec * SR].reshape(sec, SR) ** 2, axis=1) + 1e-10) if sec else np.zeros(1)
    lo, hi = max(0, int(moment_offset) - 5), min(sec, int(moment_offset) + 6)
    base = max(float(np.median(rms)), 0.01)  # a mostly silent voice track would make any word look huge
    loud = float(rms[lo:hi].max() / base) if hi > lo else 1.0
    return react + min(1.0, max(0.0, (loud - 1.5) / 3)), react, loud


def analyze_windows(audio, cands, diarize=True, log=print):
    """Adds 'vocal' (reaction strength in the separated voices) and speaker labels to candidates.
    cands: [{"moment", "start", "end", "segments": [{"t", "text"}]}] (changed in place)."""
    deep = Deep()
    import onnxruntime as ort
    from sound_events import MODEL, ensure_model
    ensure_model()
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    all_prints, owners = [], []
    for i, c in enumerate(cands):
        start, dur = c["start"], c["end"] - c["start"]
        voc = deep.vocals(audio, start, dur)
        c["vocal"], c["vocal_laugh_shout"], c["vocal_loudness"] = [round(x, 2) for x in reaction_in_vocals(voc, c["moment"] - start, sess)]
        if diarize:
            for t, e in deep.voice_prints(voc):
                all_prints.append((start + t, e))
                owners.append(i)
        log(f"   deep listening {i + 1}/{len(cands)} ({c['moment'] // 3600:02.0f}:{c['moment'] % 3600 // 60:02.0f}:{c['moment'] % 60:02.0f})")
    if diarize and all_prints:
        labels = cluster_speakers(all_prints)
        names = "ABCDEFGH"
        for i, c in enumerate(cands):
            mine = [(t, labels[j]) for j, (t, _) in enumerate(all_prints) if owners[j] == i]
            speakers = sorted({lab for _, lab in mine})
            turns = sum(1 for a, b in zip(mine, mine[1:]) if a[1] != b[1])
            c["speakers"] = len(speakers)
            c["turns_per_min"] = round(turns / max(1e-6, (c["end"] - c["start"]) / 60), 1)
            for s in c["segments"]:
                near = [lab for t, lab in mine if s["t"] - 0.5 <= t <= s["t"] + 3]
                if near and len(speakers) > 1:
                    s["speaker"] = names[max(set(near), key=near.count) % len(names)]
    return cands
