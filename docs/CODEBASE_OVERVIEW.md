# Codebase overview: training pipeline and Hugging Face data flow

This document ties together the main scripts, `training/` modules, and how examples move from the Hub to the model and metrics.

## Configuration layers

| File | Role |
|------|------|
| **Training YAML** (e.g. `configs/example_mlp.yaml`, `configs/strong_baseline_knn_mp.yaml`) | Chooses `train.model`, optimizer settings, `paths.data_split` / `paths.eval_protocol`, batch size, point subsampling, checkpoints, MLflow experiment name. |
| **`configs/data_split.yaml`** | Dataset repo id, optional revision, `layout`, split mode (`hf_native` / `hash_ids` / `explicit_lists`), `id_key`, master `seed`, and split parameters. Version is logged as `data_split_version`. |
| **`configs/eval_protocol.yaml`** | KPI definitions, `eval_subsample_N`, `primary_kpi`, `version` → logged as `eval_protocol_version`. |

Training resolves the device with `training.device_utils.resolve_train_device`: if you set `train.device: cuda` or `mps` but that backend is unavailable (e.g. CPU-only PyTorch on macOS), a warning is emitted and **CPU** is used.

## End-to-end flow (training)

1. **`scripts/train.py`** loads the training YAML, then `data_split` and `eval_protocol` YAMLs, seeds from `data_split.seed`, and builds `torch.device` via `resolve_train_device`.
2. **`models.registry.get_model_class`** instantiates the model with `config={"skip_weights": True, ...}` merged from `train.model_config`.
3. **Epoch mode** runs when `train.max_epochs` is set: `_run_epoch_training` loops `train_one_epoch` → validation → early stopping / checkpoints. **Step mode** (legacy) runs when `max_epochs` is absent: `_run_legacy_step_training` with `max_train_steps` (overridable by CLI `--max-train-steps`).
4. **`training.epoch_loop`** drives each phase: it pulls batches from **`training.hf_dataset.streaming_batches`**, runs `model(t, pos, idcs_airfoil, velocity_in)`, computes losses/metrics via **`training.metrics`**, and logs to MLflow. Checkpoints go to `checkpoints/` (paths from YAML).

## Hugging Face data path

### Opening a stream

`training.hf_dataset.build_stream(data_split_path, phase)`:

1. Reads **`configs/data_split.yaml`**: `dataset.repo_id`, optional `revision`, `layout` (`auto` \| `hub_table` \| `npz`), split block, `id_key`, `seed`.
2. Chooses the Hub split name (e.g. `hf_native` named splits, or a single split string for `hash_ids` / `explicit_lists`).
3. **Row iterator** (`_resolve_row_iter`):
   - **`layout: hub_table`** or **`auto`** with a loadable table: `datasets.load_dataset(repo, split=…, streaming=True)` and `iter(ds)`.
   - **`layout: npz`** or **`auto`** falling back when the Hub repo only exposes `.npz` shards: `training.hf_npz_hub.iter_npz_root_samples` yields one dict per sample.
4. **Split filtering** (when not `hf_native`):
   - **`hash_ids`**: for each row, `training.split_assign.assign_phase_hash_ids` assigns train/val/test from a stable hash of `id_key` (see `docs/SPLIT_CONFIGURATION.md`).
   - **`explicit_lists`**: keep rows whose id is in the phase’s id set (`training.explicit_ids`).

### From row dict to `Batch`

`streaming_batches(...)` wraps `build_stream` and:

1. For each row, **`row_to_tensors`** maps fields `t`, `pos`, `velocity_in`, `velocity_out`, `idcs_airfoil` to tensors on the training **device**.
2. Optionally **subsamples points** with the same index set for positions and velocities and remaps surface indices (`subsample_points_inplace` or stratified lam/turb mode via `train.point_subsample` in YAML).
3. Buffers samples until `batch_size` is reached, then **`stack_batch`** → a **`training.hf_dataset.Batch`** (`t`, `pos`, `idcs_airfoil` as a list per sample, `velocity_in`, `velocity_out`, optional `lam_point_mask`).

### Evaluation-time point count (e.g. kNN)

If `train.eval_preforward_subsample_N` is set, **`subsample_batch_preforward`** (in `hf_dataset.py`) can reduce points **before** `forward` on val/test so graph-based models see a fixed point count consistent with training subsampling.

## Evaluation and leaderboard

| Entry point | Behavior |
|-------------|----------|
| **`scripts/eval_checkpoint.py`** | Loads a `.pt` state dict with architecture from the same training YAML, runs the **test** split via **`training.eval_runner.evaluate_checkpoint_on_test`**, logs `test/*` to MLflow. Uses `resolve_train_device` when no device override is passed. |
| **`scripts/leaderboard.py`** | Reads a manifest (see `configs/leaderboard_manifest.example.yaml`), evaluates multiple checkpoints on the test split, ranks by a metric (default `test/l2_per_point_mean`). Implementation uses **`training.eval_runner`** and **`training/leaderboard_rank.py`**. |

## Metrics and MLflow keys

- **Scalar batch/epoch metrics** follow `configs/eval_protocol.yaml` (`val/l2_per_point_mean`, `val/mse_velocity`, stream keys, etc.).
- **`evaluate_split_full`** in `training/epoch_loop.py` also aggregates **per-output-timestep** means when applicable, e.g. `val/l2_timestep_{i}_mean`, `val/mse_timestep_{i}_mean`, and lam/turb variants when those KPIs are enabled for the split.

## Registered models

Names are defined in **`models/registry.py`**: `tiny_linear`, `mlp`, `strong_mlp`, `strong_mlp_knn`, `strong_mlp_knn_mp`. Adding a model is described in **`docs/ADDING_A_MODEL.md`**.

## Related docs

- **Short repo tour:** `docs/SIMPLIFIED_README.md`
- **Setup & HF login:** `docs/SETUP_NEW_MACHINE.md`
- **Split modes and `id_key`:** `docs/SPLIT_CONFIGURATION.md`
- **MLflow usage and fair comparisons:** `docs/EXPERIMENTS_AND_MLFLOW.md`
- **Submission:** `docs/SUBMISSION_CHECKLIST.md`

## Key file map

| Area | Files |
|------|--------|
| Train CLI | `scripts/train.py` |
| HF stream + collate | `training/hf_dataset.py`, `training/hf_npz_hub.py`, `training/split_assign.py`, `training/explicit_ids.py` |
| Loops + eval splits | `training/epoch_loop.py` |
| Losses / metrics | `training/metrics.py` |
| Checkpoint eval | `training/eval_runner.py`, `scripts/eval_checkpoint.py` |
| Leaderboard | `scripts/leaderboard.py`, `training/leaderboard_rank.py` |
| Device resolution | `training/device_utils.py` |
| YAML loading | `training/yaml_config.py` |
