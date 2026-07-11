import numpy as np

import generate_negatives
from audio_variants import PRESET_VOICES, build_variants
from phrases import GENERIC_NEGATIVE_SENTENCES, HARD_NEGATIVE_PHRASES


def _fake_backend_factory():
    def backend(text, voice, temperature, top_k):
        return np.zeros(8000, dtype=np.float32), 16000
    return backend


def test_main_writes_hard_and_generic_subfolders(tmp_path):
    generate_negatives.main(
        argv=["--out-dir", str(tmp_path)],
        backend_factory=_fake_backend_factory,
    )

    hard_wavs = list((tmp_path / "hard").glob("*.wav"))
    generic_wavs = list((tmp_path / "generic").glob("*.wav"))

    # Expected count computed independently of what main() actually does,
    # from the module's own reduced negative-specific axis constants plus
    # the real (full) voice count, so this doesn't just echo whatever
    # main() happens to produce.
    non_voice_combinations = (
        len(generate_negatives.NEGATIVE_TEMPERATURES)
        * len(generate_negatives.NEGATIVE_TOP_KS)
        * len(generate_negatives.NEGATIVE_PITCH_SEMITONES)
        * len(generate_negatives.NEGATIVE_SPEED_FACTORS)
    )
    expected_variant_count = len(PRESET_VOICES) * non_voice_combinations
    assert expected_variant_count == 216

    assert len(hard_wavs) == len(HARD_NEGATIVE_PHRASES) * expected_variant_count
    assert len(generic_wavs) == len(GENERIC_NEGATIVE_SENTENCES) * expected_variant_count
    assert len(hard_wavs) == 1512
    assert len(generic_wavs) == 2160

    # Sanity check: the reduced negative grid must be meaningfully smaller
    # than the full production grid used for positives (3,600 variants),
    # so negative generation doesn't balloon back into tens of thousands of
    # clips per phrase.
    full_production_grid_size = len(build_variants())
    assert full_production_grid_size == 3600
    assert expected_variant_count < full_production_grid_size

    assert all(p.name.startswith("hardneg_") for p in hard_wavs)
    assert all(p.name.startswith("genneg_") for p in generic_wavs)
