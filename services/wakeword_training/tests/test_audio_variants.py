from audio_variants import GenerationVariant, build_variants, PRESET_VOICES


def test_preset_voices_nonempty():
    assert len(PRESET_VOICES) == 6
    assert len(set(PRESET_VOICES)) == 6


def test_variant_tag_is_unique_per_combo():
    variants = build_variants(
        voices=["A", "B"],
        temperatures=[0.7, 1.0],
        top_ks=[30],
        pitch_semitones=[0],
        speed_factors=[1.0],
    )
    assert len(variants) == 4  # 2 voices * 2 temperatures * 1 * 1 * 1
    tags = {v.tag for v in variants}
    assert len(tags) == 4


def test_build_variants_default_grid_size_is_in_target_range():
    # Spec targets ~3,000-4,000 clean positive clips for a single phrase;
    # this grid times 1 phrase should land in that range.
    variants = build_variants()
    assert 3000 <= len(variants) <= 4500


def test_variant_fields_roundtrip():
    v = GenerationVariant(voice="X", temperature=1.0, top_k=50, pitch_semitones=2, speed_factor=1.1)
    assert v.voice == "X"
    assert v.temperature == 1.0
    assert v.top_k == 50
    assert v.pitch_semitones == 2
    assert v.speed_factor == 1.1
    assert "X" in v.tag
