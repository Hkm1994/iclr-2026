# Experiments and MLflow

## Start a training run

```bash
source .venv/bin/activate
export PYTHONPATH=.
python scripts/train.py --config configs/example_mlp.yaml
```

Optional: `--max-train-steps 10` for a quick smoke (overrides YAML if wired—currently overrides `max_train_steps` in script).

## What is logged

- **Params:** `mlflow_run_name`, `model_family`, `model`, `data_split_version`, `eval_protocol_version`, `seed`, lr, batch size, `train_subsample_N`, `eval_subsample_N`, config path.
- **Metrics:** `train/mse_velocity`, `val/mse_velocity`, `val/l2_per_point_mean` (keys come from `configs/eval_protocol.yaml`).

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
