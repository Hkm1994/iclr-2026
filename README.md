# Competition track of the GRaM workshop

[![](https://img.shields.io/badge/Website-GRaM_workshop-white)](https://gram-workshop.github.io)
[![](https://img.shields.io/badge/Website-GRaM_competition-teal)](https://gram-competition.github.io)
[![](https://img.shields.io/badge/Hugging_Face-Dataset-yellow)](https://huggingface.co/datasets/gram-competition/warped-ifw)

<img src=".logos/beyondmath.svg" height="60vw">&nbsp;&nbsp;&nbsp;<img src=".logos/mcml.svg" height="60vw">

This repository functions as submission portal for the competition hosted in conjunction with the Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM) at ICLR 2026.
For description of the challenge refer to the competition website (link above).

Deadline is on **April 22, 2026 (AoE)**.

## Local development (training stack)

This repo includes a small training pipeline: **Hugging Face streaming**, central **`configs/data_split.yaml`** / **`configs/eval_protocol.yaml`**, **MLflow** logging, and **pytest** for the submission contract.

- **Setup:** [docs/SETUP_NEW_MACHINE.md](docs/SETUP_NEW_MACHINE.md)
- **Splits & seeds:** [docs/SPLIT_CONFIGURATION.md](docs/SPLIT_CONFIGURATION.md)
- **New models:** [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)
- **Experiments / MLflow:** [docs/EXPERIMENTS_AND_MLFLOW.md](docs/EXPERIMENTS_AND_MLFLOW.md)
- **Pre-PR checks:** [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md)

Quick start:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
pytest -q
python scripts/train.py --config configs/example_mlp.yaml --max-train-steps 5
```

Train/eval need HF access to `gram-competition/warped-ifw`; run `python scripts/inspect_dataset.py` after `huggingface-cli login` to confirm schema and set `id_key` in `configs/data_split.yaml`.

## Submission guidelines

In order to participate in the competition, your team has to create a valid submission in the form of a **pull request** to this repository. The requirements for a valid submission are listed in the following.
You can mimic our implementation of a basic MLP for reference.

Create a **class implementation** of your model (fully contained) in the directory `models/<model name>/` that can be instantiated without arguments, i.e.,
```python
model = ModelName()
```
and is callable (e.g., via `model.forward`) by the signature
```python
def __call__(
    t: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    velocity_in: torch.Tensor
) -> torch.Tensor:
    ...
    return velocity_out
```
with tensor dimensions
```
t: (batch size, 10)
pos: (batch size, 100k, 3)
velocity_in: (batch size, 5, 100k, 3)
velocity_out: (batch size, 5, 100k, 3)
```
where elements of the list `idcs_airfoil` are variable-length tensors indexing `pos`, i.e., take values in `[0, 100k)`.

Feel free to use a different backend than PyTorch (JAX, etc.) but please match the typing in the signature above.
It is fine if your model depends on external libraries (xFormers, your own, etc.) as long as they are easy to install.

Provide **model weights** along with your pull request, either by uploading to the directory `models/<model name>/` or via download link if their file size would be too large.
Your model must load the weights during construction.

Create an **import entry** in `models/__init__.py` that imports you model's constructor from `models/<model name>/`.

*Optional:* provide a Markdown file under `models/<model name>/` detailing your training process and other important information to reproduce your approach.

## Submission policy
Each team may create one submission.
We will monitor submissions on a rolling basis and notify participants once their submission is valid.
