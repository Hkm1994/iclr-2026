# Submission checklist (internal)

Official rules: [README.md](../README.md) and [gram-competition/iclr-2026](https://github.com/gram-competition/iclr-2026).

Training pipeline reference (HF loader, eval, leaderboard): [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md).

## Organizer requirements

- [ ] `model = YourModel()` — no required constructor args.
- [ ] Weights loaded in `__init__` (local `state_dict.pt` or documented download).
- [ ] `forward` / `__call__` with `t`, `pos`, `idcs_airfoil`, `velocity_in` and output shape `(B, 5, 100k, 3)`.
- [ ] Export in `models/__init__.py` for the class you submit.
- [ ] Dependencies installable from `requirements.txt` (and extras documented if needed).

## Before opening the PR

1. Pick the best run in MLflow (`primary_kpi` from `configs/eval_protocol.yaml`; note the **official** metric is undisclosed—this is a **proxy**).
2. Copy/export `state_dict.pt` to `models/<submission_name>/`.
3. Ensure `__init__(config=None)` loads that checkpoint.
4. Run:

```bash
PYTHONPATH=. python scripts/verify_submission_contract.py --num-pos 100000
PYTHONPATH=. python main.py   # optional; slow on CPU with full batch
pytest -q
```

5. Optional: add `models/<name>/training_process.md` for reproducibility.

## Large weights

If GitHub rejects file size, use a download URL in `__init__` (organizers allow this)—document the URL and hash in the checklist PR description.
