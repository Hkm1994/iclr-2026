from pathlib import Path

from training.split_assign import assign_phase_hash_ids
from training.yaml_config import load_yaml


def test_hash_ids_stable_across_calls():
    assert assign_phase_hash_ids("sample-a", 42, 0.1, 0.05) == assign_phase_hash_ids(
        "sample-a", 42, 0.1, 0.05
    )


def test_hash_ids_assignment_changes_with_master_seed():
    phases_a = [assign_phase_hash_ids(f"id-{i}", 7, 0.1, 0.05) for i in range(80)]
    phases_b = [assign_phase_hash_ids(f"id-{i}", 8, 0.1, 0.05) for i in range(80)]
    assert phases_a != phases_b


def test_data_split_yaml_loads():
    p = Path(__file__).resolve().parents[1] / "configs" / "data_split.yaml"
    cfg = load_yaml(p)
    assert "version" in cfg
    assert cfg["split"]["mode"] in ("hf_native", "hash_ids", "explicit_lists")
