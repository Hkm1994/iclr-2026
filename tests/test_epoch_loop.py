from training.epoch_loop import is_better


def test_is_better_lower():
    assert is_better(0.9, 1.0, min_delta=0.01, lower_is_better=True)
    assert not is_better(0.99, 1.0, min_delta=0.02, lower_is_better=True)


def test_is_better_higher():
    assert is_better(1.1, 1.0, min_delta=0.01, lower_is_better=False)
    assert not is_better(1.005, 1.0, min_delta=0.01, lower_is_better=False)
