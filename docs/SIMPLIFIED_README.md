# Simplified guide: repository layout and code flow

This is a short companion to the [main README](../README.md). It skips competition rules and focuses on **where things live** and **how data moves through training**.

For more detail (HF streaming, split modes, file-level map), see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md).

---

## Repository structure (what each top-level folder is for)

```
iclr-2026/
├── README.md              # Competition + submission rules (canonical)
├── main.py                # Organizer-facing entry (batch inference contract)
├── requirements.txt       # Python deps
│
├── configs/               # YAML: training runs, data split, eval protocol, leaderboard manifest
├── models/                # One subfolder per model; registered in models/registry.py
├── training/              # Data loading, loops, metrics, MLflow helpers, device resolution
├── scripts/               # CLI: train, eval checkpoint, leaderboard, inspect dataset, etc.
├── tests/                 # pytest (contract + training components)
├── docs/                  # Setup, experiments, this guide, full pipeline overview
├── checkpoints/           # Saved state_dicts (gitignored if large)
├── mlruns/                # Default MLflow file store (local)
└── leaderboard_outputs/   # JSON from scripts/leaderboard.py (optional)
```

**Configs (typical names):** `example_mlp.yaml`, `strong_baseline*.yaml` point at `paths.data_split` and `paths.eval_protocol`, set `train.model`, hyperparameters, and checkpoint paths.

**Models:** Each package exposes an `nn.Module` with the competition `forward` signature. `models/registry.py` maps string names to classes for training.

**Training package:** Everything the scripts import—streaming from Hugging Face, batching, `train_one_epoch` / validation / test passes, logging.

---

## Code flow (training, in order)

1. **You run** `python scripts/train.py --config configs/<something>.yaml` (with `PYTHONPATH=.` or from repo root as in [SETUP_NEW_MACHINE.md](SETUP_NEW_MACHINE.md)).

2. **Load YAML:** Training config → paths to `data_split.yaml` and `eval_protocol.yaml` → read seeds, KPIs, subsampling.

3. **Device:** `training/device_utils.resolve_train_device` picks CPU / CUDA / MPS (with safe fallback if CUDA or MPS was requested but is unavailable).

4. **Model:** `models/registry.get_model_class(train.model)` builds the network with `skip_weights: True` for training.

5. **Data:** `training/hf_dataset.streaming_batches` opens a **streaming** iterator on the Hub dataset (table or `.npz` shards), filters rows into train/val/test per `data_split.yaml`, converts each row to tensors, optionally subsamples points, and stacks into batches.

6. **Loop:** `training/epoch_loop` runs forward → loss (`training/metrics`) → backward/step, validation, early stopping when `max_epochs` is set; logs metrics to **MLflow** and writes **checkpoints** under `checkpoints/`.

7. **After training:** Optional test split inside the same run; or run `scripts/eval_checkpoint.py` on a saved `.pt`; or `scripts/leaderboard.py` over several checkpoints.

---

## Other scripts (one line each)

| Script | Role |
|--------|------|
| `scripts/inspect_dataset.py` | Print Hub schema; set `id_key` in `data_split.yaml` |
| `scripts/verify_submission_contract.py` | Check `main.py` / model contract |
| `scripts/report_runs.py` | Summarize MLflow runs from the CLI |
| `scripts/init_mlp_weights.py` | Generate MLP weights if missing |

---

## Where to read next

| Topic | Doc |
|--------|-----|
| Install, HF token, pytest | [SETUP_NEW_MACHINE.md](SETUP_NEW_MACHINE.md) |
| Train/val/test split modes | [SPLIT_CONFIGURATION.md](SPLIT_CONFIGURATION.md) |
| MLflow, configs, leaderboard | [EXPERIMENTS_AND_MLFLOW.md](EXPERIMENTS_AND_MLFLOW.md) |
| Full pipeline + HF loader | [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) |
| Add a model | [ADDING_A_MODEL.md](ADDING_A_MODEL.md) |
| Before opening a PR | [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) |
