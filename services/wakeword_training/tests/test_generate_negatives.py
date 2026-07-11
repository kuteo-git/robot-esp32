import numpy as np

import generate_negatives


def _fake_backend_factory():
    def backend(text, voice, temperature, top_k):
        return np.zeros(8000, dtype=np.float32), 16000
    return backend


def test_main_writes_hard_and_generic_subfolders(tmp_path, monkeypatch):
    import audio_variants

    monkeypatch.setattr(audio_variants, "PRESET_VOICES", ["v1"])
    generate_negatives.main(
        argv=["--out-dir", str(tmp_path)],
        backend_factory=_fake_backend_factory,
    )
    from phrases import HARD_NEGATIVE_PHRASES, GENERIC_NEGATIVE_SENTENCES
    variant_count = len(audio_variants.build_variants())

    hard_wavs = list((tmp_path / "hard").glob("*.wav"))
    generic_wavs = list((tmp_path / "generic").glob("*.wav"))
    assert len(hard_wavs) == len(HARD_NEGATIVE_PHRASES) * variant_count
    assert len(generic_wavs) == len(GENERIC_NEGATIVE_SENTENCES) * variant_count
    assert all(p.name.startswith("hardneg_") for p in hard_wavs)
    assert all(p.name.startswith("genneg_") for p in generic_wavs)
