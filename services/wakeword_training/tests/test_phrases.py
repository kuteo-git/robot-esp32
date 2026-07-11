from phrases import POSITIVE_PHRASE, HARD_NEGATIVE_PHRASES, GENERIC_NEGATIVE_SENTENCES


def test_positive_phrase_is_mai_oi():
    assert POSITIVE_PHRASE == "Mai ơi"


def test_hard_negatives_nonempty_unique_and_distinct_from_positive():
    assert len(HARD_NEGATIVE_PHRASES) >= 5
    assert len(HARD_NEGATIVE_PHRASES) == len(set(HARD_NEGATIVE_PHRASES))
    assert POSITIVE_PHRASE not in HARD_NEGATIVE_PHRASES
    assert all(isinstance(p, str) and p.strip() for p in HARD_NEGATIVE_PHRASES)


def test_generic_negatives_nonempty_unique():
    assert len(GENERIC_NEGATIVE_SENTENCES) >= 5
    assert len(GENERIC_NEGATIVE_SENTENCES) == len(set(GENERIC_NEGATIVE_SENTENCES))
    assert all(isinstance(s, str) and s.strip() for s in GENERIC_NEGATIVE_SENTENCES)
