"""Unit tests for leaderboard ranking (no HF / checkpoints)."""

from __future__ import annotations

import math

from training.leaderboard_rank import sort_key_for_metric, sort_leaderboard_rows


def test_sort_key_lower_is_better():
    rows = [
        {"ok": True, "test/l2_per_point_mean": 3.0},
        {"ok": True, "test/l2_per_point_mean": 2.0},
        {"ok": True, "test/l2_per_point_mean": 4.0},
    ]
    rows.sort(key=lambda r: sort_key_for_metric(r, "test/l2_per_point_mean", True))
    assert [r["test/l2_per_point_mean"] for r in rows] == [2.0, 3.0, 4.0]


def test_sort_key_higher_is_better():
    rows = [
        {"ok": True, "score": 1.0},
        {"ok": True, "score": 3.0},
    ]
    rows.sort(key=lambda r: sort_key_for_metric(r, "score", False))
    assert [r["score"] for r in rows] == [3.0, 1.0]


def test_sort_key_nan_last_when_lower_better():
    r_ok = {"ok": True, "m": 1.0}
    r_bad = {"ok": True, "m": float("nan")}
    rows = [r_bad, r_ok]
    rows.sort(key=lambda r: sort_key_for_metric(r, "m", True))
    assert rows[0]["m"] == 1.0
    assert math.isnan(rows[1]["m"])


def test_sort_leaderboard_rows_filters_ok():
    mixed = [
        {"ok": False, "test/l2_per_point_mean": 1.0},
        {"ok": True, "test/l2_per_point_mean": 3.0},
        {"ok": True, "test/l2_per_point_mean": 2.0},
    ]
    out = sort_leaderboard_rows(mixed, "test/l2_per_point_mean", True)
    assert len(out) == 2
    assert [r["test/l2_per_point_mean"] for r in out] == [2.0, 3.0]
