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

    # Expected count computed independently of build_variants(), from the
    # known non-voice axis sizes, so this doesn't just echo whatever
    # build_variants() happens to return (which would pass even if the
    # PRESET_VOICES monkeypatch silently failed to take effect).
    patched_voice_count = 2  # len(["v1", "v2"])
    non_voice_combinations = (
        len(audio_variants.TEMPERATURES)
        * len(audio_variants.TOP_KS)
        * len(audio_variants.PITCH_SEMITONES)
        * len(audio_variants.SPEED_FACTORS)
    )
    expected_variant_count = patched_voice_count * non_voice_combinations
    assert expected_variant_count == 1200

    # 1 phrase (POSITIVE_PHRASE) * 1200 variants = 1200 files.
    assert len(wavs) == 1200

    # Sanity check: this shrunk grid must be meaningfully smaller than the
    # full production grid (6 voices -> 3600), so if the monkeypatch above
    # ever silently stops working (e.g. due to the def-time-default bug),
    # this test fails loudly instead of trivially matching build_variants().
    full_production_voice_count = 6  # PRESET_VOICES is patched above, so hardcode the real count.
    full_production_grid_size = full_production_voice_count * non_voice_combinations
    assert full_production_grid_size == 3600
    assert len(wavs) < full_production_grid_size

    assert all(p.name.startswith("pos_") for p in wavs)
