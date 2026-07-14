"""Turn the robot's own chimes into augmented hard-negative WAVs.

The R1 has no acoustic echo cancellation, so its mic hears its own chimes
(especially the 2.3s ``end_of_request`` stop chime played at session end). The
Phase-1 model was never trained against these, so it scored them as "Mai ơi" and
re-woke immediately -- the deterministic session-end/button loop the user
reported. Feeding heavily-augmented copies of the chimes in as negatives teaches
the model to reject them.

Output is 16 kHz mono PCM16, matching ``tts_generate.TARGET_SAMPLE_RATE`` and the
Android WakeWordDetector contract.
"""
from __future__ import annotations

import glob
import os

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


def _load_16k_mono(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    return audio.astype("float32")


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mimic how the device's mic hears the chime through the room: varied
    volume, cheap room reverb, a late/partial catch, and mic hiss."""
    x = x * (10 ** (rng.uniform(-12, 3) / 20))  # -12..+3 dB
    if rng.random() < 0.5:  # a few decaying reflections ~ room reverb
        ir_len = int(TARGET_SAMPLE_RATE * rng.uniform(0.05, 0.3))
        ir = np.zeros(ir_len, dtype="float32")
        ir[0] = 1.0
        for _ in range(int(rng.integers(2, 6))):
            ir[int(rng.integers(1, ir_len))] += rng.uniform(0.1, 0.5)
        x = np.convolve(x, ir)[: len(x)].astype("float32")
    if rng.random() < 0.5:  # mic catches the chime partway through
        x = x[int(rng.integers(0, max(1, len(x) // 2))):]
    x = x + rng.normal(0, rng.uniform(0.0, 0.01), len(x)).astype("float32")  # mic hiss
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    return x.astype("float32")


def gen_chime_negatives(src_dir: str, out_dir: str, variants_per_clip: int = 200, seed: int = 0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    for src in sorted(glob.glob(os.path.join(src_dir, "*.wav"))):
        base = _load_16k_mono(src)
        stem = os.path.splitext(os.path.basename(src))[0]
        for i in range(variants_per_clip):
            sf.write(
                os.path.join(out_dir, f"{stem}_{i:04d}.wav"),
                _augment(base, rng),
                TARGET_SAMPLE_RATE,
                subtype="PCM_16",
            )
            written += 1
    return written


if __name__ == "__main__":
    count = gen_chime_negatives("data/negative_vi/chimes_src", "data/negative_vi/chimes", 200)
    print(f"wrote {count} chime negatives")
