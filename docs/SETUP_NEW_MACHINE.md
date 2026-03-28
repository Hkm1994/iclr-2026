# Setup on a new machine

## Python environment

- Use **Python 3.10+**.
- Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

- Install **PyTorch** for your platform if the default wheel is wrong: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Optional graph extras

```bash
pip install -r requirements-extras.txt
```

Follow [PyG install](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if `torch-geometric` fails.

## Hugging Face

1. [Create a token](https://huggingface.co/settings/tokens) and accept the dataset terms on [gram-competition/warped-ifw](https://huggingface.co/datasets/gram-competition/warped-ifw).
2. Log in:

```bash
huggingface-cli login
# or
export HF_TOKEN=...
```

## Dataset inspection (phase 1)

After login, run:

```bash
PYTHONPATH=. python scripts/inspect_dataset.py
```

Use the printed keys and shapes to set `id_key` in `configs/data_split.yaml`.

## Weights for submission MLP

If `models/mlp/state_dict.pt` is missing:

```bash
PYTHONPATH=. python scripts/init_mlp_weights.py
```

## Sanity checks

```bash
pytest -q
PYTHONPATH=. python scripts/verify_submission_contract.py --num-pos 1024
# Full-size gate before PR:
PYTHONPATH=. python scripts/verify_submission_contract.py --num-pos 100000
```

`python main.py` uses batch size 95 and 100k points per sample; on CPU it can be slow—use a GPU or rely on `verify_submission_contract` for smoke tests.

## MLflow

```bash
export MLFLOW_TRACKING_URI=file:./mlruns   # default in train script
mlflow ui --backend-store-uri ./mlruns
```

## Secrets

Copy `.env.example` to `.env` for local variables (never commit `.env`).
