import torch

from training.metrics import (
    l2_per_point_mean,
    lam_turb_mask_for_eval_subset,
    mse_l2_lam_turb_on_subset,
    mse_velocity,
    mse_velocity_train_weighted,
    temporal_turbulence_proxy,
)


def test_mse_velocity_zero_when_equal():
    x = torch.randn(2, 5, 64, 3)
    assert mse_velocity(x, x).item() == 0.0


def test_mse_velocity_train_weighted_matches_plain_when_unweighted():
    pred = torch.randn(1, 5, 32, 3)
    tgt = torch.randn(1, 5, 32, 3)
    vi = torch.randn(1, 5, 32, 3)
    a = mse_velocity(pred, tgt)
    b = mse_velocity_train_weighted(
        pred, tgt, vi, turb_alpha=0.0, timestep_weights=None
    )
    assert torch.allclose(a, b)


def test_mse_velocity_train_weighted_timestep_weights():
    pred = torch.ones(1, 5, 4, 3)
    tgt = torch.zeros(1, 5, 4, 3)
    vi = torch.zeros(1, 5, 4, 3)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0])
    loss = mse_velocity_train_weighted(
        pred, tgt, vi, turb_alpha=0.0, timestep_weights=w
    )
    assert loss.item() > 0


def test_l2_per_point_mean_matches_manual():
    pred = torch.zeros(1, 5, 10, 3)
    tgt = torch.ones(1, 5, 10, 3)
    # L2 per vector = sqrt(3), mean over 5*10 = sqrt(3)
    expected = (3**0.5)
    got = l2_per_point_mean(pred, tgt).item()
    assert abs(got - expected) < 1e-5


def test_temporal_proxy_zero_for_constant_time_series():
    vi = torch.ones(5, 32, 3)
    s = temporal_turbulence_proxy(vi)
    assert s.shape == (32,)
    assert (s == 0).all()


def test_lam_turb_mask_and_split_mse():
    n = 20
    vi = torch.zeros(1, 5, n, 3)
    vi[0, :, :10, :] = 1.0  # zero temporal fluctuation (laminar-like proxy)
    vi[0, :, 10:, :] = torch.randn(5, 10, 3)
    idx = torch.tensor([0, 1, 2, 10, 11, 12])
    m = lam_turb_mask_for_eval_subset(vi, idx)
    pred = torch.randn(1, 5, idx.numel(), 3)
    tgt = pred.clone()
    m_lam, m_turb, _, l_lam, l_turb = mse_l2_lam_turb_on_subset(pred, tgt, m)
    assert m_lam is not None and float(m_lam) == 0.0
    assert m_turb is not None and float(m_turb) == 0.0
    assert l_lam is not None and float(l_lam) == 0.0
    assert l_turb is not None and float(l_turb) == 0.0
