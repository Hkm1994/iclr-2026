# GRaM Competition @ ICLR 2026 — Reference Summary

This note summarizes the **GRaM workshop competition track** (geometry-grounded airflow forecasting). Official sources: [competition site](https://gram-competition.github.io/), [submission repo](https://github.com/gram-competition/iclr-2026), [dataset](https://huggingface.co/datasets/gram-competition/warped-ifw).

---

## What this repository contains

This workspace holds a saved HTML copy of the competition page and a minimal `readme.md`. It is **not** the submission codebase. Implement and submit via **[gram-competition/iclr-2026](https://github.com/gram-competition/iclr-2026)**.

---

## Task (CFD + ML):

- **Physics:** Transient airflow around F1-style front-wing–like geometries (Imperial Front Wing–derived CAD). BeyondMath provided transient CFD; velocity obeys **no-slip** on the airfoil surface \(\partial\Omega\).
- **Learning goal:** A **geometry-conditioned neural operator** \(G_{\partial\Omega}\): given velocity on a fixed point cloud for the **first half** of a time window, predict velocity for the **second half**.
- **Organizers’ difficulty note:** Early times already constrain **low-frequency** flow; the main challenge is **high-frequency / turbulent** structure in the future.

---

## Prizes, publication, deadline

- **MCML Award:** 500 € for the winner.
- **Proceedings:** Description of the challenge and valid submissions; participants may opt in as co-authors.
- **Deadline:** April 22, 2026 (AoE).
- **Contact:** GitHub issues on the submission repo, or `gram.competition@proton.me`.

---

## Evaluation and submission

- **Test split:** Held-out; all released data may be used for training.
- **Metric:** Not disclosed publicly; measures accuracy / similarity of predicted vs ground-truth **3D velocity**.
- **Submission:** **One pull request per team** to [gram-competition/iclr-2026](https://github.com/gram-competition/iclr-2026). Organizers validate submissions on a rolling basis.

---

## Dataset: `gram-competition/warped-ifw` (Hugging Face)

Download from Hugging Face (may require login / accepting terms).

- **181 geometries** with transient simulation; **5 time windows** per geometry.
- Geometries: 1–3 airfoils at random relative positions and pitch (rich geometric variation).

Per-sample directory: `"<simulation_id>-<time_window_index>"`.

| Field | Shape | Description |
|--------|--------|-------------|
| `t` | `(10,)` | Time stamps for the window |
| `pos` | `(100_000, 3)` | Fixed spatial sample points |
| `idcs_airfoil` | variable, e.g. `(~8k–20k,)` | Indices into `pos` for surface points |
| `pressure` | `(10, 100_000)` | Optional extra channel |
| `velocity_in` | `(5, 100_000, 3)` | Input: first half of the window |
| `velocity_out` | `(5, 100_000, 3)` | Target: second half |

---

## Submission implementation contract (`iclr-2026` repo)

1. Add a **self-contained** model class under `models/` that:
   - Instantiates with **no arguments:** `model = ModelName()`
   - **Loads weights in `__init__`** (files under `models/` or a download link if large)
   - Implements **`__call__` / `forward`** with:

```python
def __call__(
    t: torch.Tensor,                    # (B, 10)
    pos: torch.Tensor,                  # (B, 100_000, 3)
    idcs_airfoil: list[torch.Tensor],   # length B; each 1D, values in [0, 100k)
    velocity_in: torch.Tensor,          # (B, 5, 100_000, 3)
) -> torch.Tensor:                      # (B, 5, 100_000, 3)
    ...
```

2. Register the constructor in `models/__init__.py`.
3. **Optional:** `models/.../training_process.md` for training / reproducibility.
4. Non-PyTorch backends (e.g. JAX) are allowed if tensor shapes match; dependencies should be easy to install.

The repo’s `main.py` uses dummy tensors to check output shape; it prints a **hint** metric (mean pointwise \(\ell_2\) error), not necessarily the official leaderboard metric.

**Reference baseline:** `models/mlp` — per-point MLP on `pos` + flattened `velocity_in`; the template comment suggests also using `t` and `idcs_airfoil`.

---

## Modeling ideas

- Encode geometry via **`pos`** + **`idcs_airfoil`** (GNNs on k-NN or radius graphs, geometric transformers, distance-to-surface features).
- Use **`t`** explicitly (embeddings, Fourier features, or autoregressive steps).
- Consider **residual / multi-scale** heads: coarse predictor + high-frequency correction.
- **Scale:** 100k points per sample favors subsampling, latent compression, or efficient attention / graph operators.

---

## Quick links

| Resource | URL |
|----------|-----|
| GRaM workshop | https://gram-workshop.github.io |
| Competition website | https://gram-competition.github.io |
| Submission (GitHub) | https://github.com/gram-competition/iclr-2026 |
| Dataset | https://huggingface.co/datasets/gram-competition/warped-ifw |
