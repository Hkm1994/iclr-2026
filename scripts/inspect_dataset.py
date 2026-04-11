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

import datasets
from datasets import load_dataset

from training.hf_npz_hub import (
    hub_dataset_has_only_npz_error,
    load_first_npz_row,
    load_first_npz_row_local,
)


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
    p = argparse.ArgumentParser(description="Inspect warped-ifw dataset schema and splits.")
    p.add_argument("--repo", default="gram-competition/warped-ifw")
    p.add_argument("--split", default="train", help="Split to stream one example from")
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--source",
        choices=("hub", "local"),
        default="hub",
        help="hub: Hugging Face (default). local: first .npz under --local-path (no HF).",
    )
    p.add_argument(
        "--local-path",
        default=None,
        help="Directory of .npz shards when --source local (relative to cwd if not absolute).",
    )
    args = p.parse_args()

    if args.source == "local":
        if not args.local_path:
            print("--local-path is required when --source local", file=sys.stderr)
            return 2
        try:
            row = load_first_npz_row_local(args.local_path)
        except Exception as e:
            print("Failed to load local .npz:", e, file=sys.stderr)
            return 1
        print("=== Dataset ===", "local", args.local_path)
        print("\n=== Layout ===", "Local tree .npz (first shard by sorted relpath)")
        _print_example_inspection(row)
        return 0

    print("=== Dataset ===", args.repo)
    gsn = getattr(datasets, "get_dataset_split_names", None)
    if gsn is not None:
        try:
            names = gsn(args.repo, revision=args.revision)
            print("Available split names:", names)
        except Exception as e:
            hint = ""
            if hub_dataset_has_only_npz_error(e):
                hint = " (This repo ships root-level .npz files; split names do not apply.)"
            print("Could not list split names:", e, hint, sep="")
    else:
        print("get_dataset_split_names not available in this `datasets` version.")

    row = None
    try:
        kw = {"split": args.split, "streaming": True}
        if args.revision:
            kw["revision"] = args.revision
        ds = load_dataset(args.repo, **kw)
        row = next(iter(ds))
    except Exception as e:
        if hub_dataset_has_only_npz_error(e):
            try:
                row = load_first_npz_row(args.repo, revision=args.revision)
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
    _print_example_inspection(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
