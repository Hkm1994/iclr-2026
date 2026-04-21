# LeversTailV2Submission — clean PR bundle

This folder is a **copy/paste bundle** to create a minimal submission PR.

It contains the self-contained model implementation + weights, and a checklist of the
two small repo edits required by the competition submission contract.

## What to copy into a clean branch

Copy the following paths into your clean branch (preserving paths):

- `models/levers_tail_submission/` (entire folder)

## Required small edits in the clean branch

1) Export the class in `models/__init__.py`:

```python
from .levers_tail_submission import LeversTailV2Submission
```

2) (Optional, but convenient) Register it in `models/registry.py`:

```python
from models.levers_tail_submission.model import LeversTailV2Submission

MODELS["levers_tail_v2_submission"] = LeversTailV2Submission
```

## Validate the submission

From repo root:

```bash
export PYTHONPATH=.
python scripts/verify_submission_contract.py --class-name LeversTailV2Submission --num-pos 100000 --batch-size 1
pytest -q -m "not hf"
```

