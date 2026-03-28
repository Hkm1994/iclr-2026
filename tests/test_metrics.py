import torch

from training.metrics import l2_per_point_mean, mse_velocity


def test_mse_velocity_zero_when_equal():
    x = torch.randn(2, 5, 64, 3)
    assert mse_velocity(x, x).item() == 0.0


def test_l2_per_point_mean_matches_manual():
    pred = torch.zeros(1, 5, 10, 3)
    tgt = torch.ones(1, 5, 10, 3)
    # L2 per vector = sqrt(3), mean over 5*10 = sqrt(3)
    expected = (3**0.5)
    got = l2_per_point_mean(pred, tgt).item()
    assert abs(got - expected) < 1e-5
