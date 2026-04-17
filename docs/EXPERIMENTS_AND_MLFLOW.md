# Experiments and MLflow

## Start a training run

```bash
source .venv/bin/activate
export PYTHONPATH=.
python scripts/train.py --config configs/example_mlp.yaml
```

### Training modes

- **Epoch mode** (default for strong baselines): set `train.max_epochs` in the YAML. The script runs train → val each epoch, early stopping, `best` / `last` checkpoints, and optionally a full **test** pass at the end (`train.eval_test_at_end`, default true).
- **Step mode** (legacy): omit `max_epochs` and use `train.max_train_steps` (default 100). CLI **`--max-train-steps N`** overrides `max_train_steps` for a quick smoke.

### Device

`train.device` can be `cuda`, `mps`, `cpu`, or omitted (then CUDA is used if available, else CPU). If you request CUDA or MPS but the installed PyTorch cannot use it, **`training.device_utils.resolve_train_device`** warns and uses CPU so `model.to(device)` does not crash.

## What is logged

- **Params:** `mlflow_run_name`, `model_family`, `model`, `data_split_version`, `eval_protocol_version`, `seed`, lr, batch size, `train_subsample_N`, `eval_subsample_N`, `eval_preforward_subsample_N` (if set), config path, and other keys from `scripts/train.py`.
- **Metrics:** Epoch summaries use one **global MLflow step** per batch through the run (monotonic). Keys from `configs/eval_protocol.yaml` include `val/l2_per_point_mean`, `val/mse_velocity`, `epoch/*` mirrors, `stream/*` for within-epoch curves, and `test/*` after a final test pass when enabled. **`training/epoch_loop.py`** also logs per-output-timestep aggregates when computing split summaries, e.g. `val/l2_timestep_{i}_mean` / `val/mse_timestep_{i}_mean` (and lam/turb variants when applicable).

## Compare runs

```bash
mlflow ui --backend-store-uri ./mlruns
```

CLI table:

```bash
python scripts/report_runs.py --experiment gram-warped-ifw
python scripts/report_runs.py --data-split-version 0.1.0 --model-family mlp
```

## Fair comparison

Only compare runs with the same **`eval_protocol_version`** and **`data_split_version`** (unless you intentionally migrate).

## Naming

Each training run gets a **unique** MLflow `run_name`: `{model}-{utc_timestamp}-{random}` (or `{prefix}-{model}-…` if you set `mlflow_run_name_prefix` under `train:` or `experiment:` in the YAML). The same string is logged as param `mlflow_run_name`. Add extra tags from the UI or with `mlflow.set_tag` in code if you need more structure.

## Test-only evaluation

After training, evaluate a single checkpoint on the test split (MLflow `test/*`):

```bash
python scripts/eval_checkpoint.py --config configs/<your_training>.yaml --checkpoint checkpoints/<model>_best.pt
```

Shared logic lives in `training/eval_runner.py` (also used by the leaderboard).

## Leaderboard over checkpoints

Compare several checkpoints on the same test protocol using a manifest:

```bash
python scripts/leaderboard.py --manifest configs/leaderboard_manifest.example.yaml
```

Edit the manifest to point `training_config` and `checkpoint` at each run; optional entries (e.g. `strong_mlp_knn_mp`) are commented in the example file.

## Example configs

| Config | `train.model` (must exist in `models/registry.py`) |
|--------|------------------------------------------------------|
| `configs/example_mlp.yaml` | `mlp` |
| `configs/example_tiny_linear.yaml` | `tiny_linear` |
| `configs/strong_baseline.yaml` | `strong_mlp` |
| `configs/strong_baseline_cosine.yaml`, `strong_baseline_stratified.yaml` | `strong_mlp` (schedule / subsampling variants) |
| `configs/strong_baseline_knn.yaml` | `strong_mlp_knn` |
| `configs/strong_baseline_knn_mp.yaml` | `strong_mlp_knn_mp` |
| `configs/strong_baseline_knn_mp_v2.yaml` | `strong_mlp_knn_mp_v2` |
| `configs/strong_baseline_knn_mp_v2_temporal.yaml` | `strong_mlp_knn_mp_v2_temporal` |
| `configs/strong_baseline_knn_mp_v2_temporal_tuned.yaml` | `strong_mlp_knn_mp_v2_temporal` (EMA + weighted loss) |
| `configs/strong_baseline_knn_mp_v2_levers.yaml` | `strong_mlp_knn_mp_v2` (weighted loss only) |

Other `configs/example_*.yaml` files may reference model names not yet registered; use them as templates after you add the model to `models/registry.py`. For how data reaches the model, see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md).
