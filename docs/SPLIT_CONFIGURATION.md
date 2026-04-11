# Data split configuration (`configs/data_split.yaml`)

## Local vs Hub (`dataset.source`)

- **`source: hub`** (default): Load data from Hugging Face as today (`repo_id`, optional `revision`), using `layout` (`auto` / `hub_table` / `npz`).
- **`source: local`**: Read **only** `.npz` files under **`dataset.local_path`** (recursive). No Hub download for samples; HF token not required for training. **`layout` must not be `hub_table`** (raise a clear error); use `layout: npz` or `layout: auto` (auto skips the Hub table attempt and uses local NPZ).
- **`local_path`**: Directory containing the mirror (e.g. `data/warped-ifw` or `sample_data`). Relative paths are resolved against the **current working directory**—start training from the repository root (e.g. `local_path: sample_data`).
- **`sample_id`**: For each file, the id matches the Hub convention `sample_id_from_dataset_relpath(relative_path)` so **`hash_ids` splits match** a faithful file tree mirror. Renaming or flattening files differently than the Hub will change assignments.

Populate a mirror, for example:

```bash
huggingface-cli download gram-competition/warped-ifw --repo-type dataset --local-dir data/warped-ifw
```

See also [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md#hugging-face-data-path).

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

Training may use `train_subsample_N` in the **training YAML** (not in `data_split.yaml`). When subsampling points, the same index set is applied to `pos`, `velocity_in`, `velocity_out`, and surface indices are remapped (see `training/hf_dataset.py`). Optional `train.eval_preforward_subsample_N` subsamples points again before `forward` on val/test so models that depend on graph scale (e.g. kNN) see a consistent point count.

For how rows are streamed from the Hub and turned into batches, see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md#hugging-face-data-path).
