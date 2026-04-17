# Inspecting predictions (Streamlit + CLI)

Tools to compare **model velocity predictions** to **actual** simulation labels (`velocity_out`): metrics, 2D slices, 3D error coloring, and optional GIF animation.

## Install

```bash
pip install -r requirements-extras.txt
```

This adds **streamlit**, **plotly**, **pillow**, and **scipy** on top of the base [`requirements.txt`](../requirements.txt).

## Streamlit app

From the **repository root**:

```bash
streamlit run scripts/streamlit_inspect_predictions.py
```

1. Set **Training YAML** and **Checkpoint .pt** (same as training).
2. Choose **split**: `train`, `val`, or `test` (same [`streaming_batches`](../training/hf_dataset.py) phases as training).
3. Set **buffer size K**: only the first **K** batches from the stream are loaded (random sampling is **among these K**, not uniform over the full dataset).
4. Click **Load data + run model**. The app runs **forward on all K batches once** and caches predictions so changing timestep/slice does not re-run the model.
5. Use **Output timestep k** (all `T_out` steps, typically 5), **slice** controls, **3D** view (decimated for performance), and **Build GIF** for a short animation over `k`.

**Boundary conditions:** Training minimizes MSE to **data**, not enforced no-slip. Compare **prediction vs actual** on the surface; compare **actual** to your physics expectations separately.

**Subsampling:** Point counts follow your training YAML (`train_subsample_N`, `eval_preforward_subsample_N`, stratified mode if configured). For full-resolution exploration, adjust those in YAML or use a config with `train_subsample_N: null` where appropriate.

**Parallelism:** “Parallel” here means **GPU-vectorized** forward passes; batches are processed sequentially, one tensor batch at a time.

## CLI

```bash
python scripts/inspect_model_predictions.py \
  --config configs/example_mlp.yaml \
  --checkpoint checkpoints/your_model.pt \
  --phase val \
  --max-batches 8 \
  --save-dir figures_inspect/
```

- **`--random-seed`**: pick a random buffer index (still among the first `--max-batches` batches).
- **`--device`**: `cpu`, `cuda`, or `mps` (default: auto from config).

## Phase 2 (not implemented here)

VTK export, wake/near-wall regional L2, and multi-checkpoint comparison can be added later.
