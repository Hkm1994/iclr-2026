import pytest
import torch


def pytest_configure():
    torch.manual_seed(0)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_batch_tensors():
    b, n, num_t_in, num_t_out = 2, 256, 5, 5
    t = torch.randn(b, num_t_in + num_t_out)
    pos = torch.randn(b, n, 3)
    idcs_airfoil = [
        torch.randint(0, n, (42,)),
        torch.randint(0, n, (17,)),
    ]
    velocity_in = torch.randn(b, num_t_in, n, 3)
    return t, pos, idcs_airfoil, velocity_in
