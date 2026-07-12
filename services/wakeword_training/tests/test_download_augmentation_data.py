import numpy as np
import soundfile as sf

from download_augmentation_data import _write_16k_wav


def test_write_16k_wav_writes_int16_pcm_at_16khz(tmp_path):
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    _write_16k_wav(tmp_path, "clip.wav", audio)

    path = tmp_path / "clip.wav"
    assert path.exists()
    data, sample_rate = sf.read(str(path), dtype="int16")
    assert sample_rate == 16000
    assert len(data) == 5


def test_write_16k_wav_creates_parent_directory(tmp_path):
    out_dir = tmp_path / "nested" / "dir"
    _write_16k_wav(out_dir, "clip.wav", np.zeros(10, dtype=np.float32))
    assert (out_dir / "clip.wav").exists()
