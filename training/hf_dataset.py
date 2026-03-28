"""Hugging Face streaming loader, split filtering, and collate (competition shapes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from datasets import load_dataset

from training.explicit_ids import load_id_set
from training.hf_npz_hub import hub_dataset_has_only_npz_error, iter_npz_root_samples
from training.split_assign import assign_phase_hash_ids, get_sample_id
from training.yaml_config import load_yaml

SplitPhase = Literal["train", "val", "test"]


@dataclass
class Batch:
    t: torch.Tensor  # (B, 10)
    pos: torch.Tensor  # (B, N, 3)
    idcs_airfoil: list[torch.Tensor]
    velocity_in: torch.Tensor  # (B, 5, N, 3)
    velocity_out: torch.Tensor  # (B, 5, N, 3)


def load_data_split_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def _as_float_tensor(x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _as_long_tensor(x: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)


def row_to_tensors(
    row: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Map one HF row to tensors (single sample, no batch dim yet)."""
    t = _as_float_tensor(row["t"], device, dtype)
    pos = _as_float_tensor(row["pos"], device, dtype)
    velocity_in = _as_float_tensor(row["velocity_in"], device, dtype)
    velocity_out = _as_float_tensor(row["velocity_out"], device, dtype)
    raw_idcs = row["idcs_airfoil"]
    idcs_airfoil = _as_long_tensor(raw_idcs, device)
    return {
        "t": t,
        "pos": pos,
        "idcs_airfoil": idcs_airfoil,
        "velocity_in": velocity_in,
        "velocity_out": velocity_out,
    }


def subsample_points_inplace(
    tensors: dict[str, Any],
    n_points: int | None,
    rng: np.random.Generator,
) -> None:
    """Mutate tensors dict in place: same index set for pos / velocities; remap surface idcs."""
    if n_points is None:
        return
    pos = tensors["pos"]
    n = pos.shape[0]
    if n_points >= n:
        return
    idx_np = np.sort(rng.choice(n, size=n_points, replace=False))
    idx = torch.tensor(idx_np, dtype=torch.long, device=pos.device)
    inv = {int(old): new for new, old in enumerate(idx.tolist())}

    tensors["pos"] = pos[idx]
    tensors["velocity_in"] = tensors["velocity_in"][:, idx, :]
    tensors["velocity_out"] = tensors["velocity_out"][:, idx, :]

    idcs = tensors["idcs_airfoil"]
    new_idcs = torch.tensor(
        [inv[int(i)] for i in idcs.tolist() if int(i) in inv],
        dtype=torch.long,
        device=pos.device,
    )
    tensors["idcs_airfoil"] = new_idcs


def stack_batch(samples: list[dict[str, Any]]) -> Batch:
    """Stack single-sample tensor dicts into a batch (variable idcs_airfoil as list)."""
    t = torch.stack([s["t"] for s in samples], dim=0)
    pos = torch.stack([s["pos"] for s in samples], dim=0)
    vi = torch.stack([s["velocity_in"] for s in samples], dim=0)
    vo = torch.stack([s["velocity_out"] for s in samples], dim=0)
    idcs_list = [s["idcs_airfoil"] for s in samples]
    return Batch(t=t, pos=pos, idcs_airfoil=idcs_list, velocity_in=vi, velocity_out=vo)


def _split_name_for_phase(ds_cfg: dict[str, Any], phase: SplitPhase) -> str:
    mode = ds_cfg["split"]["mode"]
    if mode == "hf_native":
        nat = ds_cfg["split"]["hf_native"]
        if phase == "train":
            return nat["train_split"]
        if phase == "val":
            return nat["val_split"]
        ts = nat.get("test_split")
        if ts is None:
            raise ValueError("test_split not configured")
        return ts
    if mode == "hash_ids":
        return ds_cfg["split"]["hash_ids"]["hf_split"]
    if mode == "explicit_lists":
        return ds_cfg["split"]["explicit_lists"]["hf_split"]
    raise ValueError(f"Unknown mode {mode}")


def _open_hf_table_stream(repo: str, split_name: str, rev: str | None):
    kw: dict[str, Any] = {"streaming": True, "split": split_name}
    if rev:
        kw["revision"] = rev
    return load_dataset(repo, **kw)


def _npz_row_stream(
    repo: str,
    rev: str | None,
    phase: SplitPhase,
    master_seed: int,
) -> Iterator[dict[str, Any]]:
    shuffle = phase == "train"
    rng = np.random.default_rng(master_seed + {"train": 11, "val": 13, "test": 17}[phase])
    yield from iter_npz_root_samples(repo, revision=rev, shuffle=shuffle, rng=rng if shuffle else None)


def build_stream(
    data_split_path: str | Path,
    phase: SplitPhase,
) -> Iterator[dict[str, Any]]:
    cfg = load_yaml(data_split_path)
    repo = cfg["dataset"]["repo_id"]
    rev = cfg["dataset"].get("revision") or None
    layout = cfg["dataset"].get("layout", "auto")
    split_block = cfg["split"]
    mode = split_block["mode"]
    id_key = cfg["id_key"]
    master_seed = int(cfg["seed"])

    def _resolve_row_iter(*, hf_split_name: str) -> Iterator[dict[str, Any]]:
        if layout == "npz":
            if mode == "hf_native":
                raise ValueError(
                    "split.mode hf_native is not supported for dataset.layout npz "
                    "(Hub repo has no named splits; use hash_ids or explicit_lists)."
                )
            return _npz_row_stream(repo, rev, phase, master_seed)
        if layout == "hub_table":
            ds = _open_hf_table_stream(repo, hf_split_name, rev)
            return iter(ds)

        # auto: try Hub table loader, then .npz-at-root layout (warped-ifw).
        if mode == "hf_native":
            ds = _open_hf_table_stream(repo, hf_split_name, rev)
            return iter(ds)
        try:
            ds = _open_hf_table_stream(repo, hf_split_name, rev)
            return iter(ds)
        except Exception as e:
            if not hub_dataset_has_only_npz_error(e):
                raise
            return _npz_row_stream(repo, rev, phase, master_seed)

    if mode == "hf_native":
        name = _split_name_for_phase(cfg, phase)
        yield from _resolve_row_iter(hf_split_name=name)
        return

    if mode == "hash_ids":
        hf_split = split_block["hash_ids"]["hf_split"]
    elif mode == "explicit_lists":
        hf_split = split_block["explicit_lists"]["hf_split"]
    else:
        raise ValueError(mode)

    row_iter = _resolve_row_iter(hf_split_name=hf_split)

    if mode == "hash_ids":
        vf = float(split_block["hash_ids"]["val_fraction"])
        tf = float(split_block["hash_ids"]["test_fraction"])
        for row in row_iter:
            sid = get_sample_id(row, id_key)
            p = assign_phase_hash_ids(sid, master_seed, vf, tf)
            if p == phase:
                yield row
        return

    if mode == "explicit_lists":
        el = split_block["explicit_lists"]
        train_ids = load_id_set(el.get("train_ids_path"))
        val_ids = load_id_set(el.get("val_ids_path"))
        test_ids = load_id_set(el.get("test_ids_path"))
        sets = {"train": train_ids, "val": val_ids, "test": test_ids}
        allowed = sets[phase]
        for row in row_iter:
            sid = get_sample_id(row, id_key)
            if sid in allowed:
                yield row
        return

    raise ValueError(f"Unknown split mode: {mode}")


def streaming_batches(
    data_split_path: str | Path,
    phase: SplitPhase,
    *,
    device: torch.device,
    batch_size: int = 1,
    train_subsample_N: int | None = None,
    point_seed: int = 0,
    dtype: torch.dtype = torch.float32,
    max_batches: int | None = None,
) -> Iterator[Batch]:
    """Yield collated batches; optional point subsampling per sample (training)."""
    stream = build_stream(data_split_path, phase)
    buf: list[dict[str, Any]] = []
    n_out = 0
    sample_counter = 0
    for row in stream:
        rng = np.random.default_rng(int(point_seed) + sample_counter)
        sample_counter += 1
        tensors = row_to_tensors(row, device=device, dtype=dtype)
        subsample_points_inplace(tensors, train_subsample_N, rng)
        buf.append(tensors)
        if len(buf) >= batch_size:
            yield stack_batch(buf)
            buf.clear()
            n_out += 1
            if max_batches is not None and n_out >= max_batches:
                return
    if buf and (max_batches is None or n_out < max_batches):
        yield stack_batch(buf)
