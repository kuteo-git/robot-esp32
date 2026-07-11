from metrics import compute_metrics


def test_all_correct():
    result = compute_metrics(positive_scores=[0.9, 0.8], negative_scores=[0.1, 0.2], threshold=0.5)
    assert result.false_reject_rate == 0.0
    assert result.false_accept_rate == 0.0
    assert result.num_false_rejects == 0
    assert result.num_false_accepts == 0


def test_all_wrong():
    result = compute_metrics(positive_scores=[0.1, 0.2], negative_scores=[0.9, 0.8], threshold=0.5)
    assert result.false_reject_rate == 1.0
    assert result.false_accept_rate == 1.0


def test_mixed():
    result = compute_metrics(positive_scores=[0.9, 0.1, 0.6], negative_scores=[0.1, 0.6, 0.2], threshold=0.5)
    assert result.num_positive == 3
    assert result.num_negative == 3
    assert result.num_false_rejects == 1  # the 0.1
    assert result.num_false_accepts == 1  # the 0.6
    assert result.false_reject_rate == 1 / 3
    assert result.false_accept_rate == 1 / 3


def test_empty_lists_do_not_divide_by_zero():
    result = compute_metrics(positive_scores=[], negative_scores=[], threshold=0.5)
    assert result.false_reject_rate == 0.0
    assert result.false_accept_rate == 0.0
    assert result.num_positive == 0
    assert result.num_negative == 0
