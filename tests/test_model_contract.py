import pytest
import torch

from models.mlp.model import MLP
from models.registry import MODELS


@pytest.mark.parametrize("name", list(MODELS.keys()))
def test_model_forward_shape_and_finite(name, small_batch_tensors):
    cls = MODELS[name]
    model = cls(config={"skip_weights": True})
    model.eval()
    t, pos, idcs_airfoil, velocity_in = small_batch_tensors
    with torch.no_grad():
        out = model(t, pos, idcs_airfoil, velocity_in)
    b, n = pos.shape[0], pos.shape[1]
    assert out.shape == (b, 5, n, 3)
    assert torch.isfinite(out).all()


def test_mlp_submission_no_arg_loads_weights_if_present():
    m = MLP()
    t, pos, idcs, vi = (
        torch.randn(1, 10),
        torch.randn(1, 32, 3),
        [torch.arange(5)],
        torch.randn(1, 5, 32, 3),
    )
    with torch.no_grad():
        o = m(t, pos, idcs, vi)
    assert o.shape == (1, 5, 32, 3)


def test_backward_smoke_mlp(small_batch_tensors):
    model = MLP(config={"skip_weights": True})
    t, pos, idcs_airfoil, velocity_in = small_batch_tensors
    pred = model(t, pos, idcs_airfoil, velocity_in)
    pred.mean().backward()
