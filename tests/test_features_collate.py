import numpy as np
import torch

from models.features import knn_indices_brute_force, surface_mask_from_idcs
from training.hf_dataset import subsample_points_inplace


def test_surface_mask():
    b, n = 2, 100
    idcs = [torch.tensor([0, 1, 2]), torch.tensor([10, 20])]
    m = surface_mask_from_idcs(b, n, idcs, device=torch.device("cpu"))
    assert m.shape == (b, n)
    assert m[0, 0] == 1 and m[0, 3] == 0
    assert m[1, 10] == 1


def test_knn_smoke():
    pos = torch.randn(20, 3)
    idx = knn_indices_brute_force(pos, k=4)
    assert idx.shape == (20, 4)


def test_subsample_remaps_surface():
    n = 500
    rng = np.random.default_rng(0)
    tensors = {
        "pos": torch.randn(n, 3),
        "velocity_in": torch.randn(5, n, 3),
        "velocity_out": torch.randn(5, n, 3),
        "idcs_airfoil": torch.arange(0, 50),
    }
    subsample_points_inplace(tensors, 128, rng)
    assert tensors["pos"].shape[0] == 128
    assert tensors["idcs_airfoil"].max() < 128
    assert tensors["velocity_in"].shape[1] == 128
