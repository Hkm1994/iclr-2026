# Data split configuration (`configs/data_split.yaml`)

## Modes

- **`hf_native`**: Use named splits from the Hub (`train_split`, `val_split`, optional `test_split`). Use when the dataset publishes splits.
- **`hash_ids`**: Stream a single HF split (`hash_ids.hf_split`) and assign each example to train/val/test using a **deterministic** SHA-256–based rule (`training/split_assign.py`). **Do not use Python `hash()`** for ids.
- **`explicit_lists`**: Provide paths to text/JSON files listing ids for each phase; stream `explicit_lists.hf_split` and filter rows by id.

## `id_key`

Must match a field in each dataset row that uniquely identifies the simulation/window. Set it after running `python scripts/inspect_dataset.py`.

## Versioning

Bump `version` whenever you change fractions, lists, or mode. The value is logged as MLflow param `data_split_version` so comparable runs use the same split.

## Organizer test set

The **competition leaderboard** uses a **held-out test** you do not receive. Any `test_split` here is **internal** only (e.g. early stopping), not the official test.

## Streaming + `hash_ids`

Filtering happens while streaming; for very large corpora you may export id lists once and switch to `explicit_lists` for faster epoch boundaries.

## Collate / point subsampling

Training may use `train_subsample_N` in the **training YAML** (not in `data_split.yaml`). When subsampling points, the same index set is applied to `pos`, `velocity_in`, `velocity_out`, and surface indices are remapped (see `training/hf_dataset.py`).
