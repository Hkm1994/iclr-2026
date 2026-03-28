#!/usr/bin/env python3
"""Print schema for gram-competition/warped-ifw (streaming). Requires HF auth + dataset access."""

from __future__ import annotations

import argparse
import sys
from pprint import pprint

import datasets
from datasets import load_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect warped-ifw dataset schema and splits.")
    p.add_argument("--repo", default="gram-competition/warped-ifw")
    p.add_argument("--split", default="train", help="Split to stream one example from")
    p.add_argument("--revision", default=None)
    args = p.parse_args()

    print("=== Dataset ===", args.repo)
    gsn = getattr(datasets, "get_dataset_split_names", None)
    if gsn is not None:
        try:
            names = gsn(args.repo, revision=args.revision)
            print("Available split names:", names)
        except Exception as e:
            print("Could not list split names:", e)
    else:
        print("get_dataset_split_names not available in this `datasets` version.")

    try:
        kw = {"split": args.split, "streaming": True}
        if args.revision:
            kw["revision"] = args.revision
        ds = load_dataset(args.repo, **kw)
    except Exception as e:
        print("Failed to load dataset:", e, file=sys.stderr)
        print("Ensure: huggingface-cli login (or HF_TOKEN) and accept dataset terms on the Hub.", file=sys.stderr)
        return 1

    it = iter(ds)
    row = next(it)
    print("\n=== One example keys ===")
    pprint(list(row.keys()))

    print("\n=== Shapes / dtypes (best effort) ===")
    import numpy as np

    for k, v in row.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k}: len={len(v)} type={type(v).__name__}")
        else:
            print(f"  {k}: type={type(v).__name__} repr={repr(v)[:120]}")

    print("\n=== Suggested id_key for configs/data_split.yaml ===")
    for cand in ("simulation_id", "id", "sample_id", "__index_level_0__"):
        if cand in row:
            print(f"  Found key {cand!r} -> set id_key: {cand}")
            break
    else:
        print("  No common id key found; pick a stable unique field from keys above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
