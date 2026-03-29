import tempfile
from pathlib import Path

import torch
from torch import nn

from training.train_checkpointing import save_state_dict_atomic, state_dict_cpu


def test_save_state_dict_atomic_roundtrip():
    m = nn.Linear(4, 2)
    m.weight.data.fill_(1.25)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "w.pt"
        save_state_dict_atomic(p, m)
        blob = torch.load(p, map_location="cpu", weights_only=True)
        assert torch.allclose(blob["weight"], state_dict_cpu(m)["weight"])
