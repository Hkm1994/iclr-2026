# Adding a model for experiments

## Layout

Create `models/<your_model>/` with:

- `model.py` — `nn.Module` implementing the competition forward contract.
- `__init__.py` — export your class.
- `state_dict.pt` — weights loaded in `__init__` for submission (or download URL logic if weights are huge).

## API contract

Match the organizer README:

- `Model()` with **no required arguments** (use `def __init__(self, config: dict | None = None)`; `config is None` = submission mode).
- `forward(t, pos, idcs_airfoil, velocity_in)` with shapes `(B,10)`, `(B,N,3)`, list of `(Ni,)`, `(B,5,N,3)` → `(B,5,N,3)`.
- Weights loaded inside `__init__` when submitting.

Training typically uses `config={"skip_weights": True, ...}` plus optimizer-driven updates; export a checkpoint to `state_dict.pt` for the PR.

## Registry

Register in `models/registry.py`:

```python
from models.your_model.model import YourModel

MODELS["your_model"] = YourModel
```

Current keys: `tiny_linear`, `mlp`, `strong_mlp`, `strong_mlp_knn`, `strong_mlp_knn_mp`, `strong_mlp_knn_mp_v2`, `strong_mlp_knn_mp_v2_temporal` (see `MODELS` in `models/registry.py`).

## Config

Add `configs/example_your_model.yaml` pointing at:

- `paths.data_split`
- `paths.eval_protocol`
- `train.model: your_model`

## Tests

Contract tests iterate `models.registry.MODELS`. If your model needs PyG, mark tests `@pytest.mark.geo` and skip when extras are missing.

## Submission export

See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for promoting the best MLflow run to a PR-ready checkpoint.

## See also

[CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) for how training pulls batches from Hugging Face and calls your `forward`.
