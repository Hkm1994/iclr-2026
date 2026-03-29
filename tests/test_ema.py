import torch

from models.registry import MODELS
from training.ema import ModelEMA


def test_model_ema_update_and_swap():
    m = MODELS["tiny_linear"](config={"skip_weights": True})
    ema = ModelEMA(m, decay=0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    backup = ema.swap_in_shadow(m)
    try:
        sd = m.state_dict()
        for k in ema.shadow:
            if torch.is_floating_point(ema.shadow[k]):
                assert not torch.allclose(sd[k], backup[k])
    finally:
        ema.restore(m, backup)
