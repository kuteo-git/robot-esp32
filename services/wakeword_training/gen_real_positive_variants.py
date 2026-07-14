"""Augment real "Mai ơi" recordings into training positives.

The synthetic TTS positives don't cover real voices/rooms well enough (~80%
detection on real recordings). This multiplies a set of real "Mai ơi" clips into
pitch/speed/gain variants so the model learns the real speakers' timbre and
prosody. RIR + background-noise augmentation is added later by
extract_features.py; this stage adds only prosodic/timbre variety.

Output is 16 kHz mono PCM16.
"""
from __future__ import annotations

import glob
import os

import librosa
import numpy as np
import soundfile as sf

from tts_generate import apply_pitch_speed

TARGET_SAMPLE_RATE = 16000


def _load_16k_mono(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    return audio.astype("float32")


def gen_real_positive_variants(src_dir: str, out_dir: str, variants_per_clip: int = 40, seed: int = 0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    for src in sorted(glob.glob(os.path.join(src_dir, "*.wav"))):
        base = _load_16k_mono(src)
        stem = os.path.splitext(os.path.basename(src))[0]
        for i in range(variants_per_clip):
            pitch = int(rng.integers(-3, 4))          # -3..+3 semitones
            speed = float(rng.uniform(0.9, 1.1))      # ±10% tempo
            x = apply_pitch_speed(base, TARGET_SAMPLE_RATE, pitch, speed)
            x = x * (10 ** (rng.uniform(-6, 6) / 20))  # ±6 dB
            peak = float(np.abs(x).max()) if x.size else 0.0
            if peak > 1.0:
                x = x / peak
            sf.write(
                os.path.join(out_dir, f"{stem}_{i:03d}.wav"),
                x.astype("float32"),
                TARGET_SAMPLE_RATE,
                subtype="PCM_16",
            )
            written += 1
    return written


if __name__ == "__main__":
    n = gen_real_positive_variants(
        "data/real_eval/positive_train_src", "data/positive/real_aug", variants_per_clip=40
    )
    print(f"wrote {n} real-positive variants")
