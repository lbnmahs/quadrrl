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

### legged_vs_wheeled_performance.ipynb

Multi-seed **legged vs wheeled** analysis for the 16-condition × 3-seed campaign (20k iterations, 1024 envs, seeds 42 / 0 / 1):

- Go2 ↔ Go2W, B2 ↔ B2W, ZSL1 ↔ ZSL1W, Lite3 ↔ M20 (flat + rough)

Loads only seed-tagged runs for those experiments. Reports terminal **mean ± std**, flat vs rough and paired morphology comparisons (wheeled − legged deltas), learning-stage snapshots / ΔR at 25/50/75/100% of the 20k budget, and convergence bands with 5k/10k/15k/20k markers. Exports CSV/LaTeX/PNG under `notebooks/exports/legged_vs_wheeled/`.

Headless smoke test (same pipeline as the notebook):

```bash
python scripts/analysis/run_legged_vs_wheeled_analysis.py
```

### Fair-Morph-v2 (comparable re-run)

Parallel Fair task IDs (`*-Fair-v0`) retrain **Go2↔Go2W**, **B2↔B2W**, **ZSL1↔ZSL1W**, and **Lite3↔M20** with matched tracking weights (`1.5` / `0.75`), wheeled `upward=0`, equal `[512,256,128]` agents with obs norm, and softer B2 rough terrain. Logs land under `logs/rsl_rl/*_fair/` (do not mix with native LVW).

Train:

```bash
bash scripts/experiments/run_fair_morph_v2_seeds.sh
```

Analyze with the same LVW notebook helpers, pointing at Fair experiments. Prefer tracking + episode length as primary metrics (`mean_reward` secondary):

```python
from rsl_rl_analysis_utils import (
    FAIR_MORPH_V2_CONDITIONS,
    FAIR_MORPH_V2_METRIC_COLUMNS,
    FAIR_MORPH_V2_PAIRS,
    build_morphology_paired_delta_dataframe,
    load_seed_tagged_metrics,
)

metrics = load_seed_tagged_metrics(
    experiments=list(FAIR_MORPH_V2_CONDITIONS),
    conditions=FAIR_MORPH_V2_CONDITIONS,
)
# Prefer FAIR_MORPH_V2_METRIC_COLUMNS (tracking + episode_length primary).
paired = build_morphology_paired_delta_dataframe(
    per_seed_df,
    metric_columns=FAIR_MORPH_V2_METRIC_COLUMNS,
    pairs=FAIR_MORPH_V2_PAIRS,
    experiment_name_template="{stem}_{terrain}_fair",
)
```

## Usage

### Prerequisites

```bash
pip install jupyter matplotlib seaborn pandas numpy tensorboard scipy
```

### Running

From the repo root (recommended so paths resolve):

```bash
jupyter notebook notebooks/rsl_rl_performance.ipynb
jupyter notebook notebooks/legged_vs_wheeled_performance.ipynb
```

Or with JupyterLab / VS Code / Cursor notebook UI — open the same file and run all cells after training.

### Train first

```bash
bash scripts/experiments/run_table1_seeds.sh
bash scripts/experiments/run_legged_vs_wheeled_seeds.sh
bash scripts/experiments/run_fair_morph_v2_seeds.sh
# optional GPU pin:
# CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_table1_seeds.sh
```

Logs land under `logs/rsl_rl/<experiment>/<timestamp>_seed{N}/`.

## Analysis Utilities

`scripts/analysis/rsl_rl_analysis_utils.py` provides:

- `TABLE1_CONDITIONS` / `COMPARISONS` — eight-condition grid (no Anymal-C Manager)
- `LEGGED_VS_WHEELED_CONDITIONS` / `MORPHOLOGY_PAIRS` — 16-condition morphology campaign
- `FAIR_MORPH_V2_CONDITIONS` / `FAIR_MORPH_V2_PAIRS` / `FAIR_MORPH_V2_METRIC_COLUMNS` — Fair Go2↔Go2W, B2↔B2W, ZSL1↔ZSL1W, Lite3↔M20 re-run
- `load_seed_tagged_metrics()` — seed-tagged TensorBoard load only
- `extract_per_seed_terminal_metrics()` / `aggregate_across_seeds()` — seed mean±std
- `build_table1_dataframe()` / `export_table1_csv()` / `export_table1_latex()`
- `snapshot_at_fractions()` / `stage_deltas_from_snapshots()` / `curve_auc()` — budget-fraction stage learning
- `build_morphology_*` / `export_morphology_*` — LVW tables under `notebooks/exports/legged_vs_wheeled/`
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
