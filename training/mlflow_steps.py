"""Separate MLflow step axes so stream metrics do not share one global counter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamStepCounters:
    """
    Length-1 lists so callees can mutate in place across batches.

    - ``train_batch``: training stream metrics (``stream/train_*``)
    - ``val_batch``: validation stream metrics (``stream/val_*``)
    - ``test_batch``: held-out test stream metrics (``stream/test_*``)

    Epoch-level summaries (``epoch/*``, ``val/*`` KPIs at end of each epoch) are logged
    from ``scripts/train.py`` using the **val stream step** after that epoch's validation
    pass so they align with ``stream/val_*`` on the MLflow x-axis. Metric
    ``epoch/epoch_index`` records the zero-based epoch number at that step.
    """

    train_batch: list[int]
    val_batch: list[int]
    test_batch: list[int]


def new_stream_counters() -> StreamStepCounters:
    return StreamStepCounters([0], [0], [0])
