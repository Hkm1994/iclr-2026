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

    Epoch-level metrics (``epoch/*``, end-of-epoch ``val/*`` summaries) use the
    zero-based **epoch index** as ``step``, not these counters.
    """

    train_batch: list[int]
    val_batch: list[int]
    test_batch: list[int]


def new_stream_counters() -> StreamStepCounters:
    return StreamStepCounters([0], [0], [0])
