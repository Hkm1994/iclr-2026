"""Unique MLflow run names (timestamp + random suffix)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _sanitize_run_slug(s: str, max_len: int) -> str:
    t = s.replace("/", "-").replace(" ", "_").strip()
    return t[:max_len] if max_len else t


def make_mlflow_run_name(
    model_name: str,
    train_cfg: dict,
    exp_cfg: dict,
) -> str:
    """
    Unique run name: optional prefix + model + UTC time + random suffix.

    Prefix from ``train.mlflow_run_name_prefix`` or
    ``experiment.mlflow_run_name_prefix`` in the training YAML.
    """
    prefix = train_cfg.get("mlflow_run_name_prefix")
    if prefix is None:
        prefix = exp_cfg.get("mlflow_run_name_prefix")
    slug = _sanitize_run_slug(str(model_name), 48)
    if prefix is not None and str(prefix).strip():
        base = f"{_sanitize_run_slug(str(prefix), 64)}-{slug}"
    else:
        base = slug
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suf = uuid.uuid4().hex[:12]
    name = f"{base}-{ts}-{suf}"
    return name[:200]
