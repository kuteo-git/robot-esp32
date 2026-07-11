from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from audio_variants import GenerationVariant

TTSBackend = Callable[[str, str, float, int], tuple[np.ndarray, int]]


TARGET_SAMPLE_RATE = 16000  # every downstream consumer (microWakeWord feature
# extraction, evaluate.py's TFLiteScorer) assumes 16kHz mono, matching the
# Android app's AudioRecorder/WakeWordDetector contract.


def make_vieneu_backend() -> TTSBackend:
    """Real backend. Must be run under services/.venv (has `vieneu` installed)."""
    from vieneu import Vieneu

    tts = Vieneu(mode="standard", emotion="natural")

    def backend(text: str, voice: str, temperature: float, top_k: int) -> tuple[np.ndarray, int]:
        voice_data = tts.get_preset_voice(voice)
        # Vieneu.infer() returns just the audio array, not (audio, sample_rate) --
        # its native rate is exposed separately via tts.sample_rate (24kHz for the
        # standard/GGUF backend, not 16kHz).
        audio = np.asarray(tts.infer(text, voice=voice_data, temperature=temperature, top_k=top_k), dtype=np.float32)
        native_sample_rate = tts.sample_rate
        if native_sample_rate != TARGET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=native_sample_rate, target_sr=TARGET_SAMPLE_RATE)
        return audio, TARGET_SAMPLE_RATE

    return backend


def apply_pitch_speed(
    audio: np.ndarray, sample_rate: int, pitch_semitones: int, speed_factor: float
) -> np.ndarray:
    out = audio
    if pitch_semitones != 0:
        out = librosa.effects.pitch_shift(out, sr=sample_rate, n_steps=pitch_semitones)
    if speed_factor != 1.0:
        out = librosa.effects.time_stretch(out, rate=speed_factor)
    return out.astype(np.float32)


def _clip_filename(label_prefix: str, text: str, variant: GenerationVariant) -> str:
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{label_prefix}_{variant.tag}_{text_hash}.wav"


def generate_dataset(
    texts: list[str],
    variants: list[GenerationVariant],
    backend: TTSBackend,
    out_dir: Path,
    label_prefix: str,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for text in texts:
        for variant in variants:
            audio, sample_rate = backend(text, variant.voice, variant.temperature, variant.top_k)
            audio = apply_pitch_speed(audio, sample_rate, variant.pitch_semitones, variant.speed_factor)
            path = out_dir / _clip_filename(label_prefix, text, variant)
            sf.write(str(path), audio, sample_rate, subtype="PCM_16")
            written.append(path)
    return written
