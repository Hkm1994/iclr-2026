#!/usr/bin/env python3
"""
Basic submission validity checks (no HF, no MLflow).

This script is intended as a lightweight pre-PR gate:
- Required files exist (model folder + weights file)
- Model class is exported from `models/__init__.py`
- No-arg construction works (weights load in __init__)
- Forward pass matches the competition tensor contract and returns finite outputs
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-path", default="models", help="Module exporting the class.")
    ap.add_argument("--class-name", required=True, help="Submission model class name.")
    ap.add_argument(
        "--weights-path",
        default=None,
        help="Optional explicit weights file to check exists (relative to repo root).",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-pos", type=int, default=100000)
    ap.add_argument("--no-fwd", action="store_true", help="Skip forward pass.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.weights_path:
        wp = (repo_root / args.weights_path).resolve()
        if not wp.is_file():
            return _fail(f"weights file not found: {wp}")

    # Check import/export
    try:
        mod = importlib.import_module(args.import_path)
    except Exception as e:
        return _fail(f"could not import {args.import_path!r}: {e}")

    if not hasattr(mod, args.class_name):
        return _fail(
            f"{args.class_name!r} not found in module {args.import_path!r} "
            "(check models/__init__.py export)"
        )

    cls = getattr(mod, args.class_name)

    # Check no-arg constructor + weight load
    try:
        model = cls()
    except Exception as e:
        return _fail(f"no-arg constructor failed for {args.class_name}: {e}")

    model.eval()

    if args.no_fwd:
        print("OK: import + no-arg construction")
        return 0

    # Forward contract
    b = int(args.batch_size)
    n = int(args.num_pos)
    t = torch.rand(b, 10)
    pos = torch.rand(b, n, 3)
    idcs_airfoil = [torch.randint(n, size=(5000,)) for _ in range(b)]
    velocity_in = torch.rand(b, 5, n, 3)

    with torch.no_grad():
        out = model(t, pos, idcs_airfoil, velocity_in)

    exp = (b, 5, n, 3)
    if tuple(out.shape) != exp:
        return _fail(f"bad output shape: got {tuple(out.shape)} expected {exp}")
    if not torch.isfinite(out).all():
        return _fail("non-finite outputs")

    print(f"OK: {args.class_name} out.shape={tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

