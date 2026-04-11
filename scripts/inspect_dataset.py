#!/usr/bin/env python3
"""Print schema for warped-ifw: Hugging Face streaming (default) or local .npz tree (--source local)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import datasets
from datasets import load_dataset

from training.hf_npz_hub import (
    hub_dataset_has_only_npz_error,
    list_npz_filenames,
    list_npz_relpaths_local,
    load_first_npz_row,
    resolve_local_dataset_root,
    row_from_npz_path,
    sample_id_from_dataset_relpath,
)
from training.yaml_config import load_yaml


def _load_split_dataset_block(data_split_path: str | None) -> tuple[dict[str, object] | None, Path | None]:
    """Return ``(dataset`` block, path) from YAML, or (None, None) if no file."""
    if data_split_path:
        p = Path(data_split_path).expanduser().resolve()
        if not p.is_file():
            print(f"--data-split not found: {p}", file=sys.stderr)
            sys.exit(2)
        cfg = load_yaml(p)
        return (cfg.get("dataset") or {}), p
    default_p = _REPO_ROOT / "configs/data_split.yaml"
    if default_p.is_file():
        cfg = load_yaml(default_p)
        return (cfg.get("dataset") or {}), default_p.resolve()
    return None, None


def _effective_dataset_args(
    *,
    args: argparse.Namespace,
    ds_block: dict[str, object] | None,
) -> tuple[str, str | None, str, str | None]:
    """Merge CLI with data_split ``dataset:`` block (CLI wins)."""
    repo = args.repo or (str(ds_block["repo_id"]) if ds_block and ds_block.get("repo_id") else None)
    repo = repo or "gram-competition/warped-ifw"

    rev = args.revision
    if rev is None and ds_block:
        r = ds_block.get("revision")
        if r is not None and str(r).strip():
            rev = str(r).strip()

    source = args.source
    if source is None:
        s = ds_block.get("source", "hub") if ds_block else "hub"
        source = str(s).strip().lower() if s is not None else "hub"
    if source not in ("hub", "local"):
        raise ValueError(f"dataset.source must be hub or local, got {source!r}")

    local_path = args.local_path
    if local_path is None and ds_block and ds_block.get("local_path") is not None:
        lp = str(ds_block.get("local_path")).strip()
        local_path = lp if lp else None

    return repo, rev, source, local_path


def _format_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n} B"


def _print_data_summary(
    row: dict,
    *,
    headline_lines: list[str] | None = None,
) -> None:
    """Aggregate stats for the loaded example plus optional dataset-level lines."""
    print("\n=== Summary ===")
    if headline_lines:
        for line in headline_lines:
            print(f"  {line}")

    tensor_bytes = 0
    print("  Fields (numpy arrays in this row):")
    for k in sorted(row.keys()):
        v = row[k]
        if not isinstance(v, np.ndarray):
            continue
        tensor_bytes += int(v.nbytes)
        print(f"    {k}: shape={v.shape} dtype={v.dtype} size={_format_bytes(int(v.nbytes))}")

    if tensor_bytes:
        print(f"  Total ndarray payload (this row): {_format_bytes(tensor_bytes)}")

    pos = row.get("pos")
    if isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[1] == 3:
        print(f"  Points per sample: {int(pos.shape[0])}")

    for name in ("velocity_in", "velocity_out"):
        v = row.get(name)
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3:
            print(f"  {name} timesteps: {int(v.shape[0])} (nodes={int(v.shape[1])})")

    t = row.get("t")
    if isinstance(t, np.ndarray) and t.ndim == 1:
        print(f"  Time samples (t): {int(t.shape[0])}")

    raw_idcs = row.get("idcs_airfoil")
    if isinstance(raw_idcs, np.ndarray):
        print(f"  Surface indices (idcs_airfoil): {int(raw_idcs.size)} entries")

    sid = row.get("sample_id")
    if sid is not None and not isinstance(sid, np.ndarray):
        print(f"  Example sample_id: {sid!r}")


def _local_dataset_headlines(root: Path, rels: list[str]) -> list[str]:
    total_disk = sum((root / rel).stat().st_size for rel in rels)
    depths = [rel.count("/") for rel in rels]
    lines = [
        f"Shards (.npz files): {len(rels)}",
        f"On-disk size (sum of files): {_format_bytes(total_disk)}",
        f"Sorted first rel: {rels[0]!r} -> sample_id={sample_id_from_dataset_relpath(rels[0])!r}",
        f"Sorted last rel: {rels[-1]!r} -> sample_id={sample_id_from_dataset_relpath(rels[-1])!r}",
        f"Nesting depth (dir levels): min={min(depths)} max={max(depths)}",
    ]
    return lines


def _hub_npz_headlines(repo: str, revision: str | None) -> list[str]:
    files = list_npz_filenames(repo, revision=revision)
    if not files:
        return ["npz shard listing:0 files (unexpected)"]
    return [
        f"Shards (.npz at repo root, Hub listing): {len(files)}",
        f"First rel: {files[0]!r} -> sample_id={sample_id_from_dataset_relpath(files[0])!r}",
        f"Last rel: {files[-1]!r} -> sample_id={sample_id_from_dataset_relpath(files[-1])!r}",
    ]


def _print_example_inspection(row: dict) -> None:
    print("\n=== One example keys ===")
    pprint(list(row.keys()))
    print("\n=== Shapes / dtypes (best effort) ===")
    for k, v in row.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k}: len={len(v)} type={type(v).__name__}")
        else:
            print(f"  {k}: type={type(v).__name__} repr={repr(v)[:120]}")
    print("\n=== Suggested id_key for configs/data_split.yaml ===")
    for cand in ("sample_id", "simulation_id", "id", "__index_level_0__"):
        if cand in row:
            print(f"  Found key {cand!r} -> set id_key: {cand}")
            break
    else:
        print("  No common id key found; pick a stable unique field from keys above.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Inspect warped-ifw dataset schema and splits. "
            "Reads configs/data_split.yaml when present (dataset.repo_id, source, local_path, revision); "
            "CLI flags override."
        )
    )
    p.add_argument(
        "--data-split",
        default=None,
        metavar="PATH",
        help="YAML with dataset block (default: <repo>/configs/data_split.yaml if it exists).",
    )
    p.add_argument(
        "--repo",
        default=None,
        help="Hub repo id (overrides data_split dataset.repo_id).",
    )
    p.add_argument("--split", default="train", help="Split to stream one example from")
    p.add_argument(
        "--revision",
        default=None,
        help="Hub revision (overrides data_split dataset.revision).",
    )
    p.add_argument(
        "--source",
        choices=("hub", "local"),
        default=None,
        help="hub | local (default: from data_split dataset.source, else hub).",
    )
    p.add_argument(
        "--local-path",
        default=None,
        help="Local .npz root (overrides data_split dataset.local_path).",
    )
    args = p.parse_args()

    ds_block, cfg_path = _load_split_dataset_block(args.data_split)
    repo, rev, source, local_path = _effective_dataset_args(args=args, ds_block=ds_block)

    if cfg_path is not None:
        print("=== data_split ===", cfg_path)

    if (
        source == "hub"
        and ds_block
        and ds_block.get("local_path") is not None
        and str(ds_block.get("local_path", "")).strip()
    ):
        print(
            "Note: data_split has dataset.local_path but dataset.source is hub, so this run uses the Hub. "
            "Set dataset.source: local (or pass --source local) to inspect that folder."
        )

    sys.stdout.flush()

    if source == "local":
        if not local_path:
            print(
                "source is local but no local_path: set dataset.local_path in data_split.yaml "
                "or pass --local-path DIR",
                file=sys.stderr,
            )
            return 2
        try:
            root = resolve_local_dataset_root(local_path)
            rels = list_npz_relpaths_local(root)
            headlines = _local_dataset_headlines(root, rels)
            path = root / rels[0]
            row = row_from_npz_path(path)
            row["sample_id"] = sample_id_from_dataset_relpath(rels[0])
        except Exception as e:
            print("Failed to load local .npz:", e, file=sys.stderr)
            return 1
        print("=== Dataset ===", "local", local_path)
        print("\n=== Layout ===", "Local tree .npz (first shard by sorted relpath)")
        _print_data_summary(row, headline_lines=headlines)
        _print_example_inspection(row)
        return 0

    print("=== Dataset ===", repo)
    gsn = getattr(datasets, "get_dataset_split_names", None)
    if gsn is not None:
        try:
            names = gsn(repo, revision=rev)
            print("Available split names:", names)
        except Exception as e:
            hint = ""
            if hub_dataset_has_only_npz_error(e):
                hint = " (This repo ships root-level .npz files; split names do not apply.)"
            print("Could not list split names:", e, hint, sep="")
    else:
        print("get_dataset_split_names not available in this `datasets` version.")

    row = None
    hub_npz_layout = False
    try:
        kw = {"split": args.split, "streaming": True}
        if rev:
            kw["revision"] = rev
        ds = load_dataset(repo, **kw)
        row = next(iter(ds))
    except Exception as e:
        if hub_dataset_has_only_npz_error(e):
            try:
                row = load_first_npz_row(repo, revision=rev)
                hub_npz_layout = True
                print("\n=== Layout ===", "Hub root .npz (loaded first shard alphabetically)")
            except Exception as e2:
                print("Failed to load .npz from Hub:", e2, file=sys.stderr)
                print(
                    "Ensure: hf auth (HF_TOKEN or huggingface-cli login) and accepted dataset terms.",
                    file=sys.stderr,
                )
                return 1
        else:
            print("Failed to load dataset:", e, file=sys.stderr)
            print(
                "Ensure: hf auth (HF_TOKEN or huggingface-cli login) and accepted dataset terms.",
                file=sys.stderr,
            )
            return 1

    assert row is not None
    headlines: list[str] | None = None
    if hub_npz_layout:
        headlines = _hub_npz_headlines(repo, rev)
    else:
        headlines = [
            "Hub table / Arrow streaming: shard count not inferred (one row shown).",
            f"Stream split: {args.split!r}",
        ]
    _print_data_summary(row, headline_lines=headlines)
    _print_example_inspection(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
