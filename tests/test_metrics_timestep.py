import torch

from training.metrics import (
    l2_per_timestep_mean,
    l2_per_timestep_mean_masked,
    mse_per_timestep_mean,
)


def test_l2_mse_per_timestep_shapes():
    b, t, n = 1, 5, 32
    pred = torch.randn(b, t, n, 3)
    tgt = torch.randn(b, t, n, 3)
    lt = l2_per_timestep_mean(pred, tgt)
    mt = mse_per_timestep_mean(pred, tgt)
    assert lt.shape == (t,)
    assert mt.shape == (t,)


def test_l2_per_timestep_masked():
    b, t, n = 1, 5, 16
    pred = torch.randn(b, t, n, 3)
    tgt = torch.randn(b, t, n, 3)
    m = torch.zeros(b, n, dtype=torch.bool)
    m[0, :8] = True
    out = l2_per_timestep_mean_masked(pred, tgt, m)
    assert out.shape == (t,)
