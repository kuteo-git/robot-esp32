"""Generate hard negatives in the robot's OWN VieNeu voice(s).

The user reported the wake word firing on the robot's own speech. The most
faithful negative is the robot's actual audio output, so this hits the running
VieNeu HTTP server (``:8002`` -- same server the robot uses, including its gain
/ tail-trim / cue post-processing) rather than the training-only standard-mode
backend in ``tts_generate.py``. The result is exactly what the mic hears.

Output is 16 kHz mono PCM16 (``TARGET_SAMPLE_RATE``), matching every downstream
consumer and the Android WakeWordDetector contract.
"""
from __future__ import annotations

import io
import os

import librosa
import numpy as np
import requests
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
VIENEU_URL = os.environ.get("VIENEU_URL", "http://127.0.0.1:8002/tts")


def _synth(text: str, voice: str) -> bytes:
    resp = requests.post(VIENEU_URL, json={"input": text, "voice": voice}, timeout=120)
    resp.raise_for_status()
    return resp.content


def _to_16k_mono(wav_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    return audio.astype("float32")


def gen_vieneu_negatives(sentences: list[str], voices: list[str], out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for vi, voice in enumerate(voices):
        for i, sentence in enumerate(sentences):
            audio = _to_16k_mono(_synth(sentence, voice))
            sf.write(
                os.path.join(out_dir, f"v{vi}_{i:04d}.wav"),
                audio,
                TARGET_SAMPLE_RATE,
                subtype="PCM_16",
            )
            written += 1
    return written


if __name__ == "__main__":
    from sentences_vi import ROBOT_NEGATIVE_SENTENCES

    # A spread across the production voice catalog (from :8002/health) so the
    # model rejects the robot regardless of which preset is active.
    VOICES = ["Ngọc Lan", "Ngọc Linh", "Gia Bảo", "Mỹ Duyên", "Trúc Ly", "Xuân Vĩnh"]
    count = gen_vieneu_negatives(ROBOT_NEGATIVE_SENTENCES, VOICES, "data/negative_vi/robot_voice")
    print(f"wrote {count} robot-voice negatives")
