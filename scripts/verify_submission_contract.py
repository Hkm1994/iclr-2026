#!/usr/bin/env python3
"""Smoke-test competition I/O: Model(), forward, output shape (default N=100k or --num-pos)."""

from __future__ import annotations

import argparse
import importlib
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--import-path",
        default="models",
        help="Module that exports the model class (default: models -> models/__init__.py)",
    )
    ap.add_argument("--class-name", default="MLP", help="Model class name")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument(
        "--num-pos",
        type=int,
        default=1024,
        help="Point cloud size (use 100000 for final pre-PR gate; default 1024 for CI/smoke)",
    )
    args = ap.parse_args()

    mod = importlib.import_module(args.import_path)
    cls = getattr(mod, args.class_name)
    model = cls()
    model.eval()

    b = args.batch_size
    n = args.num_pos
    num_t_in = 5
    num_t_out = 5
    t = torch.rand(b, num_t_in + num_t_out)
    pos = torch.rand(b, n, 3)
    idcs_airfoil = [
        torch.randint(n, size=(torch.randint(3142, 5000, (1,)).item(),))
        for _ in range(b)
    ]
    velocity_in = torch.rand(b, num_t_in, n, 3)

    with torch.no_grad():
        out = model(t, pos, idcs_airfoil, velocity_in)

    exp = (b, num_t_out, n, 3)
    if out.shape != exp:
        print(f"BAD shape: got {tuple(out.shape)} expected {exp}", file=sys.stderr)
        return 1
    if not torch.isfinite(out).all():
        print("Non-finite outputs", file=sys.stderr)
        return 1
    print(f"OK: {args.class_name} out.shape={tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
