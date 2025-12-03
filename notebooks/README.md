# Notebooks

This directory contains Jupyter notebooks for analyzing training results and visualizing performance.

## Available Notebooks

### rsl_rl_performance.ipynb

Comprehensive performance analysis focusing on **locomotion tasks**:
- Flat vs Rough terrain comparisons
- Different robot platforms (Anymal-C, Anymal-D, Unitree Go2)
- Direct locomotion vs Manager-based control
- Latest checkpoint comparisons

**Features:**
- Load metrics from TensorBoard logs
- Generate comparison bar charts
- Plot training curves
- Extract key performance metrics

## Usage

### Prerequisites

Install Jupyter and required dependencies:

```bash
pip install jupyter matplotlib seaborn pandas numpy
```

### Running Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open the desired notebook from the Jupyter interface

3. Ensure the notebook is run from the project root or adjust paths accordingly

### Notebook Structure

Notebooks typically:
1. Import analysis utilities from `scripts/analysis/`
2. Load training logs from `logs/` directory
3. Process and visualize metrics
4. Generate comparison plots

## Analysis Utilities

Notebooks use utilities from `scripts/analysis/rsl_rl_analysis_utils.py`:

- `load_all_metrics()` - Load metrics from TensorBoard logs
- `extract_key_metrics()` - Extract specific metrics
- `plot_comparison_bar()` - Generate comparison bar charts
- `plot_training_curves()` - Plot training progress over time
- `plot_multi_metric_bars()` - Multi-metric comparison plots

## Log Directory Structure

Notebooks expect logs in the following structure:

```
logs/
└── <framework>/
    └── <task_name>/
        └── <experiment_date>/
            └── events.out.tfevents.*  # TensorBoard logs
```

## Tips

1. **Update paths**: Adjust `PROJECT_ROOT` and `LOGS_DIR` if running from different locations
2. **Refresh comparisons**: Use `refresh_comparisons_with_latest()` to use latest runs
3. **TensorBoard integration**: Launch TensorBoard alongside notebooks for interactive exploration
4. **Export results**: Save plots and metrics for reports

## Related Documentation

- [Scripts Documentation](../scripts/README.md) - Analysis script details
- [Training Guide](../docs/TRAINING.md) - Understanding training outputs

