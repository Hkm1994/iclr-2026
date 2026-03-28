"""Deterministic train/val/test assignment from sample ids (no Python built-in hash())."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

SplitPhase = Literal["train", "val", "test"]


def _unit_interval_from_id(id_str: str, seed: int) -> float:
    msg = f"{seed}:{id_str}".encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_phase_hash_ids(
    sample_id: str, seed: int, val_fraction: float, test_fraction: float
) -> SplitPhase:
    u = _unit_interval_from_id(sample_id, seed)
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def get_sample_id(row: dict[str, Any], id_key: str) -> str:
    if id_key not in row:
        raise KeyError(
            f"id_key {id_key!r} not in row keys {list(row.keys())}. "
            "Run scripts/inspect_dataset.py and update configs/data_split.yaml."
        )
    v = row[id_key]
    if v is None:
        raise ValueError(f"Sample id for {id_key!r} is None")
    return str(v)


def row_matches_phase(
    row: dict[str, Any],
    id_key: str,
    phase: SplitPhase,
    split_cfg: dict[str, Any],
    master_seed: int,
) -> bool:
    mode = split_cfg["mode"]
    if mode == "hf_native":
        return True
    if mode == "hash_ids":
        h = split_cfg["hash_ids"]
        sid = get_sample_id(row, id_key)
        p = assign_phase_hash_ids(
            sid, master_seed, float(h["val_fraction"]), float(h["test_fraction"])
        )
        return p == phase
    if mode == "explicit_lists":
        raise NotImplementedError("explicit_lists filtering: load id sets in hf_dataset")
    raise ValueError(f"Unknown split mode: {mode}")
