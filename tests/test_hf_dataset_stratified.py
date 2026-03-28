import numpy as np
import torch

from training.hf_dataset import subsample_points_stratified_lam_turb_inplace


def test_stratified_subsample_shape_and_mask():
    n = 400
    n_sub = 128
    # Low fluctuation (laminar-like proxy) on first half, high on second half
    vi_lam = torch.zeros(5, n // 2, 3)
    vi_turb = torch.randn(5, n // 2, 3)
    velocity_in = torch.cat([vi_lam, vi_turb], dim=1)
    tensors = {
        "pos": torch.randn(n, 3),
        "velocity_in": velocity_in,
        "velocity_out": torch.randn(5, n, 3),
        "idcs_airfoil": torch.arange(0, 20),
    }
    rng = np.random.default_rng(42)
    subsample_points_stratified_lam_turb_inplace(
        tensors, n_sub, rng, lam_fraction=0.5, quantile=0.5
    )
    assert tensors["pos"].shape[0] == n_sub
    assert tensors["velocity_in"].shape[1] == n_sub
    assert tensors["lam_point_mask"].shape == (n_sub,)
    assert tensors["lam_point_mask"].dtype == torch.bool
    assert int(tensors["lam_point_mask"].sum()) > 0
    assert int((~tensors["lam_point_mask"]).sum()) > 0
