"""Ranking helpers for leaderboard (testable without importing scripts)."""

from __future__ import annotations

import math
from typing import Any


def sort_key_for_metric(
    row: dict[str, Any],
    metric: str,
    lower_is_better: bool,
) -> float:
    """Stable sort key; failed / missing metrics sort last when lower is better."""
    v = row.get(metric)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("inf") if lower_is_better else float("-inf")
    x = float(v)
    return x if lower_is_better else -x


def sort_leaderboard_rows(
    rows: list[dict[str, Any]],
    rank_metric: str,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    """Return only ``ok`` rows, sorted best-first by ``rank_metric``."""
    ok = [r for r in rows if r.get("ok")]
    ok.sort(key=lambda r: sort_key_for_metric(r, rank_metric, lower_is_better))
    return ok
