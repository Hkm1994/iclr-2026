import torch

from training.diagnostics_velocity import (
    bulk_mask,
    global_l2_mse,
    per_point_error_magnitude,
    surface_bulk_summary,
    worst_point_indices,
)


def test_per_point_error_magnitude():
    pred = torch.randn(2, 3, 10, 3)
    tgt = torch.randn(2, 3, 10, 3)
    e = per_point_error_magnitude(pred, tgt)
    assert e.shape == (2, 3, 10)
    assert torch.allclose(e[0, 0], (pred[0, 0] - tgt[0, 0]).norm(dim=-1))


def test_bulk_mask():
    m = bulk_mask(5, torch.tensor([1, 3]), device=torch.device("cpu"))
    assert m.sum() == 3


def test_worst_point_indices():
    err = torch.tensor([0.1, 2.0, 0.5, 1.5])
    idx, vals = worst_point_indices(err, n=2)
    assert idx.tolist() == [1, 3]
    assert torch.allclose(vals, torch.tensor([2.0, 1.5]))


def test_surface_bulk_summary():
    n = 20
    pred = torch.randn(1, 2, n, 3)
    tgt = torch.randn(1, 2, n, 3)
    idcs = [torch.tensor([0, 1, 2])]
    s = surface_bulk_summary(pred, tgt, bi=0, k=0, idcs_airfoil=idcs)
    assert s["n_surface_points"] == 3
    assert s["n_bulk_points"] == 17


def test_global_l2_mse():
    pred = torch.zeros(1, 1, 4, 3)
    tgt = torch.ones(1, 1, 4, 3)
    l2, mse = global_l2_mse(pred, tgt)
    assert abs(l2 - 3**0.5) < 1e-5
    assert abs(mse - 1.0) < 1e-5
