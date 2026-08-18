# Notebooks

This directory contains Jupyter notebooks for analyzing training results and visualizing performance.

## Available Notebooks

### rsl_rl_performance.ipynb

Multi-seed **Table I** analysis for the 8-condition RSL-RL PPO campaign (all 1500 iterations):

- Anymal-C Flat/Rough **Direct**
- Anymal-D Flat/Rough **Manager**
- Unitree Go2 Flat/Rough **Manager**
- Unitree B2 Flat/Rough **Manager**

Loads only seed-tagged runs (`*_seed{N}`, seeds 42 / 0 / 1). Reports **mean ± std across seeds** (not within-run CI), plots error-bar comparisons and convergence bands, and exports CSV/LaTeX under `notebooks/exports/`.

Direct vs Manager is framed as **C-Direct vs D-Manager** (morphologically similar), not a same-robot workflow ablation.

## Usage

### Prerequisites

```bash
pip install jupyter matplotlib seaborn pandas numpy tensorboard scipy
```

### Running

From the repo root (recommended so paths resolve):

```bash
jupyter notebook notebooks/rsl_rl_performance.ipynb
```

Or with JupyterLab / VS Code / Cursor notebook UI — open the same file and run all cells after training.

### Train first

```bash
bash scripts/experiments/run_table1_seeds.sh
# optional GPU pin:
# CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_table1_seeds.sh
```

Logs land under `logs/rsl_rl/<experiment>/<timestamp>_seed{N}/`.

## Analysis Utilities

`scripts/analysis/rsl_rl_analysis_utils.py` provides:

- `TABLE1_CONDITIONS` / `COMPARISONS` — eight-condition grid (no Anymal-C Manager)
- `load_seed_tagged_metrics()` — seed-tagged TensorBoard load only
- `extract_per_seed_terminal_metrics()` / `aggregate_across_seeds()` — seed mean±std
- `build_table1_dataframe()` / `export_table1_csv()` / `export_table1_latex()`
- `plot_seed_mean_std_bars()` / `plot_seed_curves_with_error_bands()`

## Log Directory Structure

```
logs/rsl_rl/
└── <experiment_name>/
    └── <YYYY-MM-DD_HH-MM-SS>_seed{N}/
        └── events.out.tfevents.*
```

## Related Documentation

- [Scripts Documentation](../scripts/README.md)
- [Training Guide](../docs/TRAINING.md)
