import numpy as np

import generate_negatives


def _fake_backend_factory():
    def backend(text, voice, temperature, top_k):
        return np.zeros(8000, dtype=np.float32), 16000
    return backend


def test_main_writes_hard_and_generic_subfolders(tmp_path, monkeypatch):
    # Shrink the grid so the test is fast and deterministic.
    import audio_variants

    monkeypatch.setattr(audio_variants, "PRESET_VOICES", ["v1"])
    generate_negatives.main(
        argv=["--out-dir", str(tmp_path)],
        backend_factory=_fake_backend_factory,
    )
    from phrases import HARD_NEGATIVE_PHRASES, GENERIC_NEGATIVE_SENTENCES

    hard_wavs = list((tmp_path / "hard").glob("*.wav"))
    generic_wavs = list((tmp_path / "generic").glob("*.wav"))

    # Expected count computed independently of build_variants(), from the
    # known non-voice axis sizes, so this doesn't just echo whatever
    # build_variants() happens to return (which would pass even if the
    # PRESET_VOICES monkeypatch silently failed to take effect).
    patched_voice_count = 1  # len(["v1"])
    non_voice_combinations = (
        len(audio_variants.TEMPERATURES)
        * len(audio_variants.TOP_KS)
        * len(audio_variants.PITCH_SEMITONES)
        * len(audio_variants.SPEED_FACTORS)
    )
    expected_variant_count = patched_voice_count * non_voice_combinations
    assert expected_variant_count == 600

    assert len(hard_wavs) == len(HARD_NEGATIVE_PHRASES) * expected_variant_count
    assert len(generic_wavs) == len(GENERIC_NEGATIVE_SENTENCES) * expected_variant_count
    assert len(hard_wavs) == 4200
    assert len(generic_wavs) == 6000

    # Sanity check: this shrunk grid must be meaningfully smaller than the
    # full production grid (6 voices), so if the monkeypatch above ever
    # silently stops working (e.g. due to the def-time-default bug), this
    # test fails loudly instead of trivially matching build_variants().
    full_production_voice_count = 6  # PRESET_VOICES is patched above, so hardcode the real count.
    full_production_grid_size = full_production_voice_count * non_voice_combinations
    assert full_production_grid_size == 3600
    assert expected_variant_count < full_production_grid_size

    assert all(p.name.startswith("hardneg_") for p in hard_wavs)
    assert all(p.name.startswith("genneg_") for p in generic_wavs)
