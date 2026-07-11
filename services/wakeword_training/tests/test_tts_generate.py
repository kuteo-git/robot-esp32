import numpy as np
import soundfile as sf

from audio_variants import GenerationVariant
from tts_generate import apply_pitch_speed, generate_dataset


def _fake_backend(text: str, voice: str, temperature: float, top_k: int):
    # 0.5s of silence at 16kHz, deterministic — no real TTS call.
    return np.zeros(8000, dtype=np.float32), 16000


def test_apply_pitch_speed_returns_nonempty_mono_array():
    audio = np.zeros(16000, dtype=np.float32)
    out = apply_pitch_speed(audio, sample_rate=16000, pitch_semitones=2, speed_factor=1.1)
    assert out.ndim == 1
    assert len(out) > 0


def test_generate_dataset_writes_one_wav_per_text_variant_pair(tmp_path):
    variants = [
        GenerationVariant("v1", 1.0, 50, 0, 1.0),
        GenerationVariant("v2", 1.0, 50, 0, 1.0),
    ]
    paths = generate_dataset(
        texts=["Mai ơi"],
        variants=variants,
        backend=_fake_backend,
        out_dir=tmp_path,
        label_prefix="pos",
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.suffix == ".wav"
        data, sr = sf.read(str(p))
        assert sr == 16000
        assert len(data) > 0
        assert p.name.startswith("pos_")
