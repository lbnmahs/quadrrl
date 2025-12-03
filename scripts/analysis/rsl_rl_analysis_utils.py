"""
RSL-RL Analysis Utilities

Utility functions for loading, processing, and visualizing RSL-RL training metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Try importing tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


# Comparison groups definition - LOCOMOTION TASKS ONLY.
# NOTE:
# - The timestamp field here is treated as a placeholder.
# - At runtime we automatically replace it with the **latest** run
#   found under the logs directory for each experiment using
#   ``refresh_comparisons_with_latest`` below.
COMPARISONS = {
    'flat_vs_rough': [
        ('anymal_c_flat', 'LATEST', 'Anymal-C Flat', 'flat'),
        ('anymal_c_rough', 'LATEST', 'Anymal-C Rough', 'rough'),
        # ('spot_rsl', 'LATEST', 'Spot', 'flat'),
        ('unitree_go2_flat', 'LATEST', 'Unitree Go2 Flat', 'flat'),
        ('unitree_go2_rough', 'LATEST', 'Unitree Go2 Rough', 'rough'),
        ('anymal_d_flat', 'LATEST', 'Anymal-D Flat', 'flat'),
        ('anymal_d_rough', 'LATEST', 'Anymal-D Rough', 'rough'),
    ],
    'robot_comparison_flat': [
        ('anymal_c_flat', 'LATEST', 'Anymal-C', 'anymal_c'),
        ('anymal_d_flat', 'LATEST', 'Anymal-D', 'anymal_d'),
        ('unitree_go2_flat', 'LATEST', 'Unitree Go2', 'go2'),
        # ('spot_rsl', 'LATEST', 'Spot', 'spot'),
    ],
    'robot_comparison_rough': [
        ('anymal_c_rough', 'LATEST', 'Anymal-C', 'anymal_c'),
        ('anymal_d_rough', 'LATEST', 'Anymal-D', 'anymal_d'),
        ('unitree_go2_rough', 'LATEST', 'Unitree Go2', 'go2'),
    ],
}

# Metric patterns to search for (will match actual TensorBoard names)
METRIC_PATTERNS = {
    'mean_reward': ['Train/mean_reward', 'Reward/total_reward', 'mean_reward'],
    'episode_length': ['Train/mean_episode_length', 'Episode/total_timesteps', 'Episode/length'],
    'policy_loss': ['Loss/surrogate', 'Loss/policy', 'policy_loss'],
    'value_loss': ['Loss/value_function', 'Loss/value', 'value_loss'],
    'entropy': ['Loss/entropy', 'entropy'],
    'position_tracking': ['Episode_Reward/position_tracking'],
    'orientation_tracking': ['Episode_Reward/orientation_tracking'],
    'termination_penalty': ['Episode_Reward/termination_penalty'],
    'base_contact': ['Episode_Termination/base_contact'],
    'track_lin_vel': ['Episode_Reward/track_lin_vel_xy_exp'],
    'track_ang_vel': ['Episode_Reward/track_ang_vel_z_exp'],
    'error_pos': ['Metrics/pose_command/error_pos_2d', 'Metrics/pose_command/error_pos'],
}

KEY_METRICS = list(METRIC_PATTERNS.keys())


def find_latest_run_timestamp(logs_dir: Path, exp_name: str) -> Optional[str]:
    """Return the latest run timestamp directory for an experiment.

    This scans ``logs_dir/exp_name`` and returns the lexicographically
    latest directory name, which works for the YYYY-MM-DD_HH-MM-SS naming
    convention used in this project.
    """
    exp_dir = logs_dir / exp_name
    if not exp_dir.exists():
        return None

    subdirs = [d.name for d in exp_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    return sorted(subdirs)[-1]


def refresh_comparisons_with_latest(logs_dir: Path) -> None:
    """Update ``COMPARISONS`` in-place to point to the latest runs.

    For every experiment listed in ``COMPARISONS``, this function finds the
    newest timestamp subdirectory under ``logs_dir / experiment`` and
    replaces the placeholder timestamp (e.g. ``'LATEST'``) with that value.

    This allows notebooks and scripts to automatically pick up new runs
    just by re-running the analysis cells, without hard-coding timestamps.
    """
    global COMPARISONS

    updated: Dict[str, List[tuple]] = {}
    for group_name, runs in COMPARISONS.items():
        new_runs = []
        for exp_name, _ts, display_name, category in runs:
            latest_ts = find_latest_run_timestamp(logs_dir, exp_name)
            if latest_ts is None:
                print(
                    f"[refresh_comparisons_with_latest] "
                    f"No runs found for experiment '{exp_name}' in {logs_dir}"
                )
                continue
            new_runs.append((exp_name, latest_ts, display_name, category))

        updated[group_name] = new_runs

    COMPARISONS = updated


def load_tensorboard_metrics(log_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all scalar metrics from a TensorBoard log directory."""
    if not TENSORBOARD_AVAILABLE:
        return {}

    metrics = {}

    try:
        ea = event_accumulator.EventAccumulator(
            str(log_dir),
            size_guidance={
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()

        scalar_tags = ea.Tags()['scalars']

        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            df = pd.DataFrame([
                {
                    'step': event.step,
                    'wall_time': event.wall_time,
                    'value': event.value
                }
                for event in scalar_events
            ])
            metrics[tag] = df

    except Exception as e:
        print(f"Error loading metrics from {log_dir}: {e}")
        return {}

    return metrics


def find_metric_name(metrics: Dict[str, pd.DataFrame], patterns: List[str]) -> Optional[str]:
    """Find a metric by trying multiple pattern matches."""
    for pattern in patterns:
        # Exact match
        if pattern in metrics:
            return pattern
        # Case-insensitive match
        for tag in metrics.keys():
            if tag.lower() == pattern.lower():
                return tag
        # Contains match
        for tag in metrics.keys():
            if pattern.lower() in tag.lower():
                return tag
    return None


def get_latest_checkpoint_value(metrics: Dict[str, pd.DataFrame], metric_name: str) -> Optional[float]:
    """Get the latest (final) value of a metric."""
    if metric_name is None or metric_name not in metrics:
        return None

    df = metrics[metric_name]
    if len(df) == 0:
        return None

    return float(df['value'].iloc[-1])


def get_metric_trajectory(metrics: Dict[str, pd.DataFrame], metric_name: str) -> Optional[pd.DataFrame]:
    """Get the full trajectory of a metric."""
    if metric_name not in metrics:
        return None

    return metrics[metric_name].copy()


def load_all_metrics(logs_dir: Path) -> Dict:
    """Load metrics for all runs in comparison groups."""
    all_metrics = {}

    for group_name, runs in COMPARISONS.items():
        for exp_name, timestamp, display_name, category in runs:
            run_key = f"{exp_name}/{timestamp}"
            if run_key in all_metrics:
                continue

            log_dir = logs_dir / exp_name / timestamp

            if not log_dir.exists():
                continue

            metrics = load_tensorboard_metrics(log_dir)

            if metrics:
                all_metrics[run_key] = {
                    'metrics': metrics,
                    'display_name': display_name,
                    'experiment': exp_name,
                    'timestamp': timestamp,
                    'category': category
                }

    return all_metrics


def extract_key_metrics(all_metrics: Dict) -> pd.DataFrame:
    """Extract key metrics for all runs into a DataFrame."""
    rows = []

    for run_key, run_data in all_metrics.items():
        metrics = run_data['metrics']

        row = {
            'run_key': run_key,
            'display_name': run_data['display_name'],
            'experiment': run_data['experiment'],
            'category': run_data['category'],
        }

        # Extract metrics using pattern matching
        for metric_key, patterns in METRIC_PATTERNS.items():
            metric_name = find_metric_name(metrics, patterns)
            value = get_latest_checkpoint_value(metrics, metric_name)
            row[metric_key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def plot_comparison_bar(
    metrics_df: pd.DataFrame,
    comparison_group: str,
    metric_column: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """Plot bar chart comparing a metric across runs in a comparison group."""
    if comparison_group not in COMPARISONS:
        print(f"Unknown comparison group: {comparison_group}")
        return ax

    runs = COMPARISONS[comparison_group]
    run_keys = [f"{exp}/{ts}" for exp, ts, _, _ in runs]

    comparison_data = metrics_df[metrics_df['run_key'].isin(run_keys)].copy()

    if len(comparison_data) == 0:
        print(f"No data found for comparison group: {comparison_group}")
        return ax

    if metric_column not in comparison_data.columns:
        print(f"Metric column '{metric_column}' not found.")
        return ax

    if 'category' in comparison_data.columns:
        comparison_data = comparison_data.sort_values('category')

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Handle NaN/None values - replace with 0 for plotting
    comparison_data = comparison_data.copy()
    # Convert to numeric, handling both None and NaN
    comparison_data[metric_column] = pd.to_numeric(comparison_data[metric_column], errors='coerce')
    comparison_data[metric_column] = comparison_data[metric_column].fillna(0)

    # Filter out rows with no valid data if all are NaN/None
    if (comparison_data[metric_column] == 0).all():
        print(f"Warning: All values for {metric_column} are NaN/None for comparison group {comparison_group}")
        print("  This metric may not be available for these runs.")

    x_pos = np.arange(len(comparison_data))
    values = comparison_data[metric_column].values

    # Ensure all values are numeric (convert any remaining None/NaN to 0.0)
    # This explicitly handles None, NaN, and ensures all values are floats
    values = np.array([0.0 if (v is None or pd.isna(v) or (isinstance(v, float) and np.isnan(v))) else float(v) for v in values], dtype=np.float64)

    if 'category' in comparison_data.columns:
        colors = sns.color_palette("husl", len(comparison_data['category'].unique()))
        category_colors = dict(zip(comparison_data['category'].unique(), colors))
        bar_colors = [category_colors[cat] for cat in comparison_data['category']]
    else:
        bar_colors = sns.color_palette("husl", len(comparison_data))

    # Final safety check: ensure all values are valid floats (not None/NaN)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.array([0.0 if not np.isfinite(v) else float(v) for v in values])

    bars = ax.bar(x_pos, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Run', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel or metric_column, fontsize=12, fontweight='bold')
    ax.set_title(title or f"Comparison: {metric_column}", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_data['display_name'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, values)):
        if pd.notna(val) and val != 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9)
        elif val == 0 or pd.isna(val):
            # Label missing data with N/A
            ax.text(bar.get_x() + bar.get_width() / 2., 0.01,
                    'N/A',
                    ha='center', va='bottom', fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    return ax


def plot_training_curves(
    all_metrics: Dict,
    comparison_group: str,
    metric_name: str,
    title: str = None,
    smoothing: int = 1,
    ax: plt.Axes = None
) -> plt.Axes:
    """Plot training curves for a metric across runs in a comparison group.

    metric_name can be either:
    - Exact TensorBoard metric name (e.g., 'Train/mean_reward')
    - Pattern key from METRIC_PATTERNS (e.g., 'mean_reward')
    """
    if comparison_group not in COMPARISONS:
        print(f"Unknown comparison group: {comparison_group}")
        return ax

    runs = COMPARISONS[comparison_group]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Try to find the metric using pattern matching if needed
    for exp_name, timestamp, display_name, category in runs:
        run_key = f"{exp_name}/{timestamp}"

        if run_key not in all_metrics:
            continue

        run_data = all_metrics[run_key]
        metrics = run_data['metrics']

        # Check if metric_name is a pattern key, if so find the actual metric
        actual_metric_name = metric_name
        if metric_name not in metrics and metric_name in METRIC_PATTERNS:
            # Try pattern matching
            actual_metric_name = find_metric_name(metrics, METRIC_PATTERNS[metric_name])

        if actual_metric_name is None or actual_metric_name not in metrics:
            print(f"Warning: Metric '{metric_name}' not found for {display_name}")
            continue

        df = metrics[actual_metric_name].copy()

        if smoothing > 1 and len(df) > smoothing:
            df['value'] = df['value'].rolling(window=smoothing, center=True).mean()

        ax.plot(df['step'], df['value'], label=display_name, linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(title or f"Training Curves: {metric_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_multi_metric_bars(
    metrics_df: pd.DataFrame,
    comparison_group: str,
    metric_columns: List[str],
    title: Optional[str] = None,
    normalize: bool = False
) -> None:
    """Plot grouped bar chart comparing multiple metrics across runs.

    This is better than radar charts for comparing multiple metrics as it allows
    for easier value reading and direct comparison across runs and metrics.
    """
    if comparison_group not in COMPARISONS:
        print(f"Unknown comparison group: {comparison_group}")
        return

    runs = COMPARISONS[comparison_group]
    run_keys = [f"{exp}/{ts}" for exp, ts, _, _ in runs]

    comparison_data = metrics_df[metrics_df['run_key'].isin(run_keys)].copy()

    if len(comparison_data) == 0:
        print(f"No data found for comparison group: {comparison_group}")
        return

    # Filter to available metrics (exclude non-numeric columns)
    available_metrics = []
    for m in metric_columns:
        if m in comparison_data.columns:
            # Check if column has any non-null values
            if comparison_data[m].notna().any():
                available_metrics.append(m)

    if len(available_metrics) == 0:
        print(f"None of the requested metrics found. Available: {list(comparison_data.columns)}")
        return

    # Prepare data for grouped bar chart
    # Each run will have a group of bars (one per metric)
    num_runs = len(comparison_data)
    num_metrics = len(available_metrics)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, num_runs * 2), 7))

    # Calculate bar positions
    x = np.arange(num_runs)
    width = 0.8 / num_metrics  # Width of each bar group

    # Get colors for metrics
    colors = sns.color_palette("husl", num_metrics)

    # Plot bars for each metric
    for i, metric in enumerate(available_metrics):
        values = []
        for _, row in comparison_data.iterrows():
            val = row[metric]
            if pd.isna(val) or val is None:
                values.append(0.0)
            else:
                values.append(float(val))

        # Normalize if requested
        if normalize:
            values_array = np.array(values)
            if values_array.max() > values_array.min():
                values = ((values_array - values_array.min())
                          / (values_array.max() - values_array.min())).tolist()

        # Calculate bar positions (offset for grouping)
        offset = (i - num_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=metric.replace('_', ' ').title(),
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if pd.notna(val) and val != 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{val:.2f}' if abs(val) < 100 else f'{val:.0f}',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=8, rotation=90 if abs(val) > 100 else 0)

    # Customize plot
    ax.set_xlabel('Run', fontsize=12, fontweight='bold')
    ylabel = 'Normalized Value (0-1)' if normalize else 'Metric Value'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title or f"Multi-Metric Comparison: {comparison_group.replace('_', ' ').title()}",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_data['display_name'].tolist(), rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()
