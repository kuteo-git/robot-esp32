"""Generate extra "Mai ơi" positives across the production VieNeu voice catalog.

The v4 model over-fit to one recording session of the real wake word and one small
TTS voice set, so live "Mai ơi" at varied conditions scored low. This widens voice
diversity by synthesizing the wake phrase in all 10 production voices (via the live
:8002 server) and pitch/speed-augmenting each. Feature extraction adds RIR +
background-noise (incl. the Vietnamese-speech background) on top.

16 kHz mono PCM16 out.
"""
from __future__ import annotations

import io
import os

import librosa
import numpy as np
import requests
import soundfile as sf

from tts_generate import apply_pitch_speed

TARGET_SAMPLE_RATE = 16000
VIENEU_URL = os.environ.get("VIENEU_URL", "http://127.0.0.1:8002/tts")

# From :8002/health. The robot speaks in these, and they span male/female/young/old.
VOICES = [
    "Ngọc Lan", "Gia Bảo", "Thái Sơn", "Đức Trí", "Mỹ Duyên",
    "Trúc Ly", "Xuân Vĩnh", "Trọng Hữu", "Bình An", "Ngọc Linh",
]
# A few natural framings of the wake phrase so it isn't one fixed utterance per voice.
PHRASES = ["Na Bi ơi", "Na Bi ơi!", "Ê Na Bi ơi", "Na Bi ơi, ", "Na Bi ơi ơi"]


def _synth(text: str, voice: str) -> bytes:
    r = requests.post(VIENEU_URL, json={"input": text, "voice": voice}, timeout=120)
    r.raise_for_status()
    return r.content


def _to_16k_mono(wav_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    return audio.astype("float32")


def gen_vieneu_positives(out_dir: str, variants_per_clip: int = 8, seed: int = 0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    for vi, voice in enumerate(VOICES):
        for pi, phrase in enumerate(PHRASES):
            base = _to_16k_mono(_synth(phrase, voice))
            for k in range(variants_per_clip):
                pitch = int(rng.integers(-3, 4))
                speed = float(rng.uniform(0.9, 1.12))
                x = apply_pitch_speed(base, TARGET_SAMPLE_RATE, pitch, speed)
                x = x * (10 ** (rng.uniform(-6, 6) / 20))
                peak = float(np.abs(x).max()) if x.size else 0.0
                if peak > 1.0:
                    x = x / peak
                sf.write(os.path.join(out_dir, f"v{vi}_p{pi}_{k:02d}.wav"), x.astype("float32"),
                         TARGET_SAMPLE_RATE, subtype="PCM_16")
                written += 1
    return written


if __name__ == "__main__":
    n = gen_vieneu_positives("data/positive/vieneu_prod", variants_per_clip=60)
    print(f"wrote {n} production-voice positives")
