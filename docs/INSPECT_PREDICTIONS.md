# Inspecting predictions (Streamlit + CLI)

Tools to compare **model velocity predictions** to **actual** simulation labels (`velocity_out`): metrics, 2D slices, interactive **Plotly 3D** views (decimated point clouds), and optional GIF animation.

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

### Setup (sidebar)

1. **Training config** — Dropdown of `configs/*.yaml` files that look like training entrypoints. These are **omitted** from the list (they are still valid YAML, just not shown): `data_split.yaml`, `eval_protocol.yaml`, `leaderboard_manifest.example.yaml`. If no matching files are found, a text field appears instead.
2. **Checkpoint** — Dropdown of all `**/*.pt` files under `checkpoints/` (including subfolders). If the directory is missing or empty, a text field appears so you can paste any path.
3. **Split**: `train`, `val`, or `test` (same [`streaming_batches`](../training/hf_dataset.py) phases as training).
4. **Buffer size K**: only the first **K** batches from the stream are loaded (random sampling is **among these K**, not uniform over the full dataset).
5. Click **Load data + run model**. The app runs **forward on all K batches once** and caches predictions so changing timestep / slice / 3D options does not re-run the model (until you load again).

**Boundary conditions:** Training minimizes MSE to **data**, not enforced no-slip. Compare **prediction vs actual** on the surface; compare **actual** to your physics expectations separately.

**Subsampling:** Point counts follow your training YAML (`train_subsample_N`, `eval_preforward_subsample_N`, stratified mode if configured). For full-resolution exploration, adjust those in YAML or use a config with `train_subsample_N: null` where appropriate.

**Parallelism:** “Parallel” here means **GPU-vectorized** forward passes; batches are processed sequentially, one tensor batch at a time.

**Editing `training/` while Streamlit is running:** The app reloads `training.inspect_plotly_3d` on each run so 3D helper changes apply without restarting the server. Other `training.*` modules still behave like a normal long-lived Python process—restart Streamlit if you change them and see stale behavior.

### Main view

1. **Output timestep k** — all `T_out` steps (typically 5).
2. **2D slices (matplotlib)** — plane normal and coordinate; prediction |v|, actual |v|, and error on the slice.
3. **3D (Plotly)** — see [3D visualization](#3d-visualization-plotly) below.
4. **Histograms** — surface |v| and global error distribution.
5. **Build GIF** — short animation over `k` for the current slice.

### 3D visualization (Plotly)

Implementation: [`training/inspect_plotly_3d.py`](../training/inspect_plotly_3d.py). The bulk cloud is **decimated** (max points from sidebar) for performance. **Airfoil surface nodes are excluded** from the metric-colored scatter so the colormap is not “painted” on the body; the airfoil is drawn in a **second trace** with a fixed **magenta** fill and dark edge so it stays distinct from Turbo / Inferno / Viridis / Plasma.

**Layouts**

- **Single cloud (pick color metric)** — one 3D scatter; color encodes the chosen quantity.
- **Side-by-side |v| pred vs actual** — two panels with a **shared color scale** on speed magnitude.

**Color metrics (single cloud)**

| Mode | Quantity | When it helps |
|------|-----------|----------------|
| **Relative error** | ‖Δv‖₂ / (\|v_actual\| + ε) | Default when speed varies a lot (stagnation vs jet); avoids slow regions dominating purely by small \|v\|. |
| **Absolute error** | ‖Δv‖₂ | Raw vector error everywhere. |
| **Speed magnitude error** | \|\|v_pred\| − \|v_actual\|\| | Too fast/slow only; ignores direction mismatch. |
| **\|v\| prediction** / **actual** | \|v\| | Inspect speed field shape vs labels. |

Optional **log₁₀** coloring and **p1–p99 color cap** (reduce outlier washout) apply to the display scale; see below for the metric floor.

**Metric threshold filter**

Enable **“Only show bulk points with metric ≥ threshold”** and set a **linear** cutoff (same units as the color quantity **before** log₁₀). Only bulk points passing the filter are drawn; the airfoil overlay is unchanged. For **side-by-side \|v\|**, the filter uses max(\|v_pred\|, \|v_actual\|) per point so both panels stay aligned.

**3D decimate seed** — changes which bulk points are kept (reproducible subsample).

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

The CLI does not expose the full Streamlit 3D UI; it uses the same diagnostics core (`training/diagnostics_velocity.py`, `training/inspect_viz.py`) for summaries and static figures.

## Related modules

| Module | Role |
|--------|------|
| [`training/inspect_plotly_3d.py`](../training/inspect_plotly_3d.py) | Plotly 3D scatter helpers: decimation, color modes, airfoil overlay, metric floor, dual \|v\| panels. |
| [`training/inspect_viz.py`](../training/inspect_viz.py) | Matplotlib slices, decimation helper, GIF frames. |
| [`training/inspect_predictions_common.py`](../training/inspect_predictions_common.py) | Load checkpoint, buffer batches, forward pass. |
| [`training/diagnostics_velocity.py`](../training/diagnostics_velocity.py) | Per-point errors, percentiles, surface vs bulk summaries. |

## Phase 2 (not implemented here)

VTK export, wake/near-wall regional L2, and multi-checkpoint comparison can be added later.
