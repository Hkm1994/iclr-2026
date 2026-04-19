"""Levers-tail training YAML matches model output timesteps and stays separate from levers."""

from __future__ import annotations

from pathlib import Path

from training.yaml_config import load_yaml


def test_levers_tail_yaml_loads_and_weights_length():
    root = Path(__file__).resolve().parents[1]
    p = root / "configs" / "strong_baseline_knn_mp_v2_levers_tail.yaml"
    cfg = load_yaml(p)
    assert cfg["train"]["model"] == "strong_mlp_knn_mp_v2"
    assert cfg["train"]["model_family"] == "strong_mlp_knn_mp_v2_levers_tail"
    w = cfg["train"]["loss_timestep_weights"]
    assert len(w) == 5
    assert w[-1] > w[0]
    assert "levers_tail" in cfg["train"]["best_checkpoint_path"]


def test_levers_yaml_unchanged_tail_weights():
    root = Path(__file__).resolve().parents[1]
    levers = load_yaml(root / "configs" / "strong_baseline_knn_mp_v2_levers.yaml")
    tail = load_yaml(root / "configs" / "strong_baseline_knn_mp_v2_levers_tail.yaml")
    assert levers["train"]["loss_timestep_weights"] == [0.9, 0.95, 1.0, 1.05, 1.1]
    assert tail["train"]["loss_timestep_weights"] == [0.85, 0.9, 1.0, 1.15, 1.35]
