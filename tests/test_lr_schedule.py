import torch
from torch.nn import Linear
from torch.optim import Adam

from training.lr_schedule import build_epoch_lr_scheduler, step_lr_scheduler


def test_plateau_reduces_after_flat_metric():
    m = Linear(2, 2)
    opt = Adam(m.parameters(), lr=0.1)
    train_cfg = {
        "lr_schedule": {
            "type": "plateau",
            "patience": 0,
            "factor": 0.5,
            "mode": "min",
            "threshold": 0.0,
        }
    }
    sch, kind = build_epoch_lr_scheduler(opt, train_cfg, max_epochs=10)
    assert kind == "plateau"
    lr0 = opt.param_groups[0]["lr"]
    step_lr_scheduler(sch, kind, monitor_value=1.0)
    step_lr_scheduler(sch, kind, monitor_value=1.0)
    assert opt.param_groups[0]["lr"] < lr0


def test_cosine_without_warmup():
    m = Linear(2, 2)
    opt = Adam(m.parameters(), lr=0.1)
    train_cfg = {"lr_schedule": {"type": "cosine", "eta_min": 0.0, "t_max_epochs": 3}}
    sch, kind = build_epoch_lr_scheduler(opt, train_cfg, max_epochs=3)
    assert kind == "cosine"
    for _ in range(3):
        step_lr_scheduler(sch, kind, monitor_value=0.0)
    assert opt.param_groups[0]["lr"] == 0.0


def test_warmup_then_cosine_sequential():
    m = Linear(2, 2)
    opt = Adam(m.parameters(), lr=0.1)
    train_cfg = {
        "lr_warmup_epochs": 2,
        "lr_schedule": {"type": "cosine", "eta_min": 0.0, "t_max_epochs": 10},
    }
    sch, kind = build_epoch_lr_scheduler(opt, train_cfg, max_epochs=5)
    assert kind == "sequential"
    for _ in range(5):
        step_lr_scheduler(sch, kind, monitor_value=0.0)
    assert opt.param_groups[0]["lr"] >= 0.0
