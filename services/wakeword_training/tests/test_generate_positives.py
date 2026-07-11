import numpy as np

import generate_positives


def _fake_backend_factory():
    def backend(text, voice, temperature, top_k):
        return np.zeros(8000, dtype=np.float32), 16000
    return backend


def test_main_writes_one_file_per_variant(tmp_path, monkeypatch):
    # Shrink the grid so the test is fast and deterministic.
    import audio_variants

    monkeypatch.setattr(audio_variants, "PRESET_VOICES", ["v1", "v2"])
    generate_positives.main(
        argv=["--out-dir", str(tmp_path)],
        backend_factory=_fake_backend_factory,
    )
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == len(audio_variants.build_variants())
    assert all(p.name.startswith("pos_") for p in wavs)
