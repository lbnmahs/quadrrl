# Copyright (c) 2024-2025, Laban Njoroge Mahihu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL Analysis Utilities

Utility functions for loading, processing, and visualizing RSL-RL training metrics.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
from scipy import stats
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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

# Seed tag in run folder names: ``..._seed42`` or containing ``seed0`` / ``seed1`` / ``seed42``.
_SEED_TAG_RE = re.compile(r"(?:^|[_-])seed(\d+)(?:$|[_-])", re.IGNORECASE)
_SEED_FALLBACK_RE = re.compile(r"seed(\d+)", re.IGNORECASE)

# Six-condition Table I grid (Anymal-C Direct, Anymal-D Manager, Go2 Manager).
TABLE1_CONDITIONS: Dict[str, Dict[str, str]] = {
    "anymal_c_flat_direct": {
        "display_name": "Anymal-C Flat Direct",
        "robot": "anymal_c",
        "terrain": "flat",
        "workflow": "direct",
        "category": "flat_direct",
    },
    "anymal_c_rough_direct": {
        "display_name": "Anymal-C Rough Direct",
        "robot": "anymal_c",
        "terrain": "rough",
        "workflow": "direct",
        "category": "rough_direct",
    },
    "anymal_d_flat": {
        "display_name": "Anymal-D Flat Manager",
        "robot": "anymal_d",
        "terrain": "flat",
        "workflow": "manager",
        "category": "flat",
    },
    "anymal_d_rough": {
        "display_name": "Anymal-D Rough Manager",
        "robot": "anymal_d",
        "terrain": "rough",
        "workflow": "manager",
        "category": "rough",
    },
    "unitree_go2_flat": {
        "display_name": "Unitree Go2 Flat",
        "robot": "go2",
        "terrain": "flat",
        "workflow": "manager",
        "category": "flat",
    },
    "unitree_go2_rough": {
        "display_name": "Unitree Go2 Rough",
        "robot": "go2",
        "terrain": "rough",
        "workflow": "manager",
        "category": "rough",
    },
    "unitree_b2_flat": {
        "display_name": "Unitree B2 Flat",
        "robot": "b2",
        "terrain": "flat",
        "workflow": "manager",
        "category": "flat",
    },
    "unitree_b2_rough": {
        "display_name": "Unitree B2 Rough",
        "robot": "b2",
        "terrain": "rough",
        "workflow": "manager",
        "category": "rough",
    },
}

# Terminal metrics used for Table I / seed aggregation.
TABLE1_METRIC_COLUMNS = [
    "mean_reward",
    "track_lin_vel",
    "track_ang_vel",
    "episode_length",
    "steps_to_75pct",
]

# Comparison groups — 8-condition grid (no Anymal-C Manager).
# Timestamp field is a placeholder; use seed-tagged loading for Table I.
# ``refresh_comparisons_with_latest`` still resolves LATEST for single-run plots.
COMPARISONS = {
    "flat_vs_rough": [
        ("anymal_c_flat_direct", "LATEST", "Anymal-C Flat Direct", "flat_direct"),
        ("anymal_c_rough_direct", "LATEST", "Anymal-C Rough Direct", "rough_direct"),
        ("anymal_d_flat", "LATEST", "Anymal-D Flat Manager", "flat"),
        ("anymal_d_rough", "LATEST", "Anymal-D Rough Manager", "rough"),
        ("unitree_go2_flat", "LATEST", "Unitree Go2 Flat", "flat"),
        ("unitree_go2_rough", "LATEST", "Unitree Go2 Rough", "rough"),
        ("unitree_b2_flat", "LATEST", "Unitree B2 Flat", "flat"),
        ("unitree_b2_rough", "LATEST", "Unitree B2 Rough", "rough"),
    ],
    "robot_comparison_flat": [
        ("anymal_c_flat_direct", "LATEST", "Anymal-C Direct", "anymal_c_direct"),
        ("anymal_d_flat", "LATEST", "Anymal-D Manager", "anymal_d"),
        ("unitree_go2_flat", "LATEST", "Unitree Go2", "go2"),
        ("unitree_b2_flat", "LATEST", "Unitree B2", "b2"),
    ],
    "robot_comparison_rough": [
        ("anymal_c_rough_direct", "LATEST", "Anymal-C Direct", "anymal_c_direct"),
        ("anymal_d_rough", "LATEST", "Anymal-D Manager", "anymal_d"),
        ("unitree_go2_rough", "LATEST", "Unitree Go2", "go2"),
        ("unitree_b2_rough", "LATEST", "Unitree B2", "b2"),
    ],
    # Morphologically similar: Anymal-C Direct vs Anymal-D Manager (not same-robot ablation).
    "direct_vs_manager": [
        ("anymal_c_flat_direct", "LATEST", "Anymal-C Flat Direct", "flat_direct"),
        ("anymal_d_flat", "LATEST", "Anymal-D Flat Manager", "flat_manager"),
        ("anymal_c_rough_direct", "LATEST", "Anymal-C Rough Direct", "rough_direct"),
        ("anymal_d_rough", "LATEST", "Anymal-D Rough Manager", "rough_manager"),
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


def compute_confidence_intervals(
    all_metrics: Dict,
    metric_name: str = 'mean_reward',
    window_size: int = 100
) -> pd.DataFrame:
    """Compute confidence intervals from the last N steps of training.
    
    Args:
        all_metrics: Dictionary of all loaded metrics
        metric_name: Name of the metric to compute CIs for
        window_size: Number of last steps to use for CI computation
        
    Returns:
        DataFrame with confidence interval data for each run
    """
    ci_data = []
    
    for run_key, run_data in all_metrics.items():
        metrics = run_data['metrics']
        metric_tag = find_metric_name(metrics, METRIC_PATTERNS.get(metric_name, [metric_name]))
        
        if metric_tag is None or metric_tag not in metrics:
            continue
            
        df = metrics[metric_tag].copy()
        if df.empty:
            continue
            
        # Get last window_size steps
        df = df.sort_values('step')
        last_window = df.tail(window_size)
        
        if len(last_window) > 0:
            mean_val = last_window['value'].mean()
            std_val = last_window['value'].std()
            n = len(last_window)
            # 95% confidence interval
            ci_95 = stats.t.interval(0.95, n - 1, loc=mean_val, scale=stats.sem(last_window['value']))
            
            ci_data.append({
                'run_key': run_key,
                'display_name': run_data['display_name'],
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_95[0],
                'ci_upper': ci_95[1],
                'n_samples': n
            })
    
    return pd.DataFrame(ci_data)


def get_trajectory_data(
    all_metrics: Dict,
    run_key: str,
    metric_name: str = 'mean_reward',
    window_size: int = 100
) -> Optional[np.ndarray]:
    """Get trajectory data for statistical testing.
    
    Args:
        all_metrics: Dictionary of all loaded metrics
        run_key: Key identifying the run
        metric_name: Name of the metric to extract
        window_size: Number of last steps to return
        
    Returns:
        Array of metric values from the last window_size steps, or None if not available
    """
    run_data = all_metrics.get(run_key)
    if run_data is None:
        return None
    
    metrics = run_data['metrics']
    metric_tag = find_metric_name(metrics, METRIC_PATTERNS.get(metric_name, [metric_name]))
    
    if metric_tag is None or metric_tag not in metrics:
        return None
    
    df = metrics[metric_tag].copy()
    if df.empty:
        return None
    
    df = df.sort_values('step')
    return df.tail(window_size)['value'].values


def compute_convergence_metrics(
    all_metrics: Dict,
    metric_name: str = 'mean_reward'
) -> pd.DataFrame:
    """Compute convergence rate metrics for each run.
    
    Args:
        all_metrics: Dictionary of all loaded metrics
        metric_name: Name of the metric to analyze
        
    Returns:
        DataFrame with convergence metrics for each run
    """
    convergence_data = []
    
    for run_key, run_data in all_metrics.items():
        metrics = run_data['metrics']
        metric_tag = find_metric_name(metrics, METRIC_PATTERNS.get(metric_name, [metric_name]))
        
        if metric_tag is None or metric_tag not in metrics:
            continue
        
        df = metrics[metric_tag].copy()
        if df.empty or len(df) < 10:
            continue
        
        df = df.sort_values('step')
        
        # Split into early, middle, late phases
        n = len(df)
        early = df.iloc[:n // 3]
        middle = df.iloc[n // 3:2 * n // 3]
        late = df.iloc[2 * n // 3:]
        
        # Compute slopes (rate of improvement) using linear regression
        def compute_slope(phase_df):
            if len(phase_df) < 2:
                return 0.0
            x = phase_df['step'].values
            y = phase_df['value'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return slope
        
        early_slope = compute_slope(early)
        middle_slope = compute_slope(middle)
        late_slope = compute_slope(late)
        
        # Overall convergence rate
        overall_slope = compute_slope(df)
        
        # Steps to reach 50%, 75%, 90% of final performance
        final_value = df['value'].iloc[-1]
        initial_value = df['value'].iloc[0]
        if final_value > initial_value:
            target_50 = initial_value + 0.5 * (final_value - initial_value)
            target_75 = initial_value + 0.75 * (final_value - initial_value)
            target_90 = initial_value + 0.9 * (final_value - initial_value)
            
            steps_50 = df[df['value'] >= target_50]['step'].iloc[0] if len(df[df['value'] >= target_50]) > 0 else np.nan
            steps_75 = df[df['value'] >= target_75]['step'].iloc[0] if len(df[df['value'] >= target_75]) > 0 else np.nan
            steps_90 = df[df['value'] >= target_90]['step'].iloc[0] if len(df[df['value'] >= target_90]) > 0 else np.nan
        else:
            steps_50 = steps_75 = steps_90 = np.nan
        
        convergence_data.append({
            'run_key': run_key,
            'display_name': run_data['display_name'],
            'experiment': run_data['experiment'],
            'category': run_data['category'],
            'early_slope': early_slope,
            'middle_slope': middle_slope,
            'late_slope': late_slope,
            'overall_slope': overall_slope,
            'steps_to_50pct': steps_50,
            'steps_to_75pct': steps_75,
            'steps_to_90pct': steps_90,
            'final_value': final_value,
            'initial_value': initial_value,
            'total_improvement': final_value - initial_value
        })
    
    return pd.DataFrame(convergence_data)


def extract_terminal_metrics(
    all_metrics: Dict,
    metric_name: str = "mean_reward",
    window_size: int = 100,
) -> pd.DataFrame:
    """Extract terminal-window metric statistics for each run.

    This avoids relying on a single last-point value when ranking runs.
    """
    rows = []
    for run_key, run_data in all_metrics.items():
        metrics = run_data["metrics"]
        metric_tag = find_metric_name(metrics, METRIC_PATTERNS.get(metric_name, [metric_name]))
        if metric_tag is None or metric_tag not in metrics:
            continue
        df = metrics[metric_tag].sort_values("step")
        if df.empty:
            continue
        tail = df.tail(window_size)
        rows.append(
            {
                "run_key": run_key,
                "display_name": run_data["display_name"],
                "experiment": run_data["experiment"],
                "category": run_data["category"],
                "metric_name": metric_name,
                "metric_tag": metric_tag,
                "terminal_mean": float(tail["value"].mean()),
                "terminal_std": float(tail["value"].std()) if len(tail) > 1 else 0.0,
                "terminal_count": int(len(tail)),
            }
        )
    return pd.DataFrame(rows)


def rank_runs_by_task(
    metrics_df: pd.DataFrame,
    score_columns: List[str],
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """Create per-category rankings from selected metric columns.

    The function normalizes each column within a category, applies direction,
    and computes an average composite score.
    """
    if metrics_df.empty:
        return pd.DataFrame([])
    if higher_is_better is None:
        higher_is_better = {col: True for col in score_columns}

    rows = []
    for category, category_df in metrics_df.groupby("category"):
        working = category_df.copy()
        norm_cols = []
        for col in score_columns:
            if col not in working.columns:
                continue
            vals = pd.to_numeric(working[col], errors="coerce")
            valid = vals.dropna()
            norm_col = f"{col}_norm"
            if valid.empty:
                working[norm_col] = np.nan
            elif valid.max() == valid.min():
                working[norm_col] = 0.5
            else:
                working[norm_col] = (vals - valid.min()) / (valid.max() - valid.min())
            if not higher_is_better.get(col, True):
                working[norm_col] = 1.0 - working[norm_col]
            norm_cols.append(norm_col)
        if not norm_cols:
            continue
        working["composite_score"] = working[norm_cols].mean(axis=1, skipna=True)
        working["rank"] = working["composite_score"].rank(method="dense", ascending=False, na_option="bottom")
        rows.append(working)
    if not rows:
        return pd.DataFrame([])
    return pd.concat(rows, ignore_index=True)


def plot_comparison_bar_interactive(
    metrics_df: pd.DataFrame,
    score_column: str,
    color_column: str = "category",
    title: Optional[str] = None,
):
    """Create an interactive Plotly bar chart for cross-run comparison."""
    if not PLOTLY_AVAILABLE:
        print("Plotly is not available. Install with: pip install plotly")
        return None
    if metrics_df.empty or score_column not in metrics_df.columns:
        return None
    figure = px.bar(
        metrics_df,
        x="display_name",
        y=score_column,
        color=color_column if color_column in metrics_df.columns else None,
        hover_data=["run_key", "experiment"] if "run_key" in metrics_df.columns else None,
        title=title or f"Interactive comparison: {score_column}",
    )
    figure.update_layout(xaxis_title="Run", yaxis_title=score_column)
    return figure


# ---------------------------------------------------------------------------
# Across-seed aggregation (Table I: mean ± std over seeds, not within-run CI)
# ---------------------------------------------------------------------------


def parse_seed_from_run_name(run_dir_name: str) -> Optional[int]:
    """Extract seed integer from a run folder name if it is seed-tagged.

    Accepts names like ``2026-08-10_12-00-00_seed42`` or any path segment
    containing ``seed0`` / ``seed1`` / ``seed42``.
    """
    match = _SEED_TAG_RE.search(run_dir_name)
    if match is None:
        match = _SEED_FALLBACK_RE.search(run_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def is_seed_tagged_run(run_dir_name: str) -> bool:
    """Return True if the run directory name includes a seed tag."""
    return parse_seed_from_run_name(run_dir_name) is not None


def list_seed_runs(
    logs_dir: Path,
    exp_name: str,
    seeds: Optional[Sequence[int]] = None,
) -> List[Tuple[str, int]]:
    """List seed-tagged run folders for an experiment.

    Returns ``(run_dir_name, seed)`` pairs sorted by seed then name.
    If ``seeds`` is given, only those seed values are kept. When multiple
    folders share a seed, the lexicographically latest name is kept.
    """
    exp_dir = logs_dir / exp_name
    if not exp_dir.exists():
        return []

    by_seed: Dict[int, str] = {}
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        seed = parse_seed_from_run_name(run_dir.name)
        if seed is None:
            continue
        if seeds is not None and seed not in seeds:
            continue
        prev = by_seed.get(seed)
        if prev is None or run_dir.name > prev:
            by_seed[seed] = run_dir.name

    return sorted(((name, seed) for seed, name in by_seed.items()), key=lambda x: (x[1], x[0]))


def load_seed_tagged_metrics(
    logs_dir: Path,
    experiments: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict:
    """Load TensorBoard metrics for seed-tagged runs of Table I conditions.

    Only runs whose folder names contain a seed tag are loaded. Untagged
    legacy logs are ignored so pre-campaign runs are not mixed into mean±std.
    """
    if experiments is None:
        experiments = list(TABLE1_CONDITIONS.keys())

    all_metrics: Dict = {}
    for exp_name in experiments:
        meta = TABLE1_CONDITIONS.get(exp_name, {})
        display_name = meta.get("display_name", exp_name)
        category = meta.get("category", "")
        robot = meta.get("robot", "")
        terrain = meta.get("terrain", "")
        workflow = meta.get("workflow", "")

        for run_name, seed in list_seed_runs(logs_dir, exp_name, seeds=seeds):
            log_dir = logs_dir / exp_name / run_name
            metrics = load_tensorboard_metrics(log_dir)
            if not metrics:
                continue
            run_key = f"{exp_name}/{run_name}"
            all_metrics[run_key] = {
                "metrics": metrics,
                "display_name": display_name,
                "experiment": exp_name,
                "timestamp": run_name,
                "category": category,
                "seed": seed,
                "robot": robot,
                "terrain": terrain,
                "workflow": workflow,
            }
    return all_metrics


def extract_per_seed_terminal_metrics(
    all_metrics: Dict,
    metric_keys: Optional[Sequence[str]] = None,
    include_convergence: bool = True,
) -> pd.DataFrame:
    """One row per seed run with terminal (final) metric values.

    Optionally merges ``steps_to_75pct`` from ``compute_convergence_metrics``.
    """
    if metric_keys is None:
        metric_keys = [m for m in TABLE1_METRIC_COLUMNS if m != "steps_to_75pct"]

    rows = []
    for run_key, run_data in all_metrics.items():
        metrics = run_data["metrics"]
        row = {
            "run_key": run_key,
            "display_name": run_data["display_name"],
            "experiment": run_data["experiment"],
            "category": run_data.get("category", ""),
            "seed": run_data.get("seed"),
            "robot": run_data.get("robot", ""),
            "terrain": run_data.get("terrain", ""),
            "workflow": run_data.get("workflow", ""),
            "timestamp": run_data.get("timestamp", ""),
        }
        for metric_key in metric_keys:
            patterns = METRIC_PATTERNS.get(metric_key, [metric_key])
            metric_name = find_metric_name(metrics, patterns)
            row[metric_key] = get_latest_checkpoint_value(metrics, metric_name)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if include_convergence:
        conv = compute_convergence_metrics(all_metrics, metric_name="mean_reward")
        if not conv.empty and "steps_to_75pct" in conv.columns:
            df = df.merge(
                conv[["run_key", "steps_to_75pct"]],
                on="run_key",
                how="left",
            )
    return df


def aggregate_across_seeds(
    per_seed_df: pd.DataFrame,
    metric_columns: Optional[Sequence[str]] = None,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate terminal metrics across seeds: mean ± std (sample std).

    Table I must use this seed-level std, not within-run last-window CI.
    """
    if per_seed_df.empty:
        return pd.DataFrame()

    if metric_columns is None:
        metric_columns = [c for c in TABLE1_METRIC_COLUMNS if c in per_seed_df.columns]
    if group_cols is None:
        group_cols = [
            c
            for c in ("experiment", "display_name", "category", "robot", "terrain", "workflow")
            if c in per_seed_df.columns
        ]

    rows = []
    for keys, group in per_seed_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        row["seeds"] = (
            sorted(group["seed"].dropna().unique().tolist()) if "seed" in group.columns else []
        )
        for col in metric_columns:
            if col not in group.columns:
                continue
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            if vals.empty:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_mean_std"] = "—"
            else:
                mean_val = float(vals.mean())
                std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{col}_mean"] = mean_val
                row[f"{col}_std"] = std_val
                row[f"{col}_mean_std"] = format_mean_std(mean_val, std_val)
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    """Format a value as ``mean ± std`` for tables / LaTeX."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def build_table1_dataframe(
    aggregated_df: pd.DataFrame,
    metric_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a Table I–ready DataFrame with ``mean ± std`` display columns."""
    if aggregated_df.empty:
        return pd.DataFrame()

    if metric_columns is None:
        metric_columns = [
            c.replace("_mean_std", "")
            for c in aggregated_df.columns
            if c.endswith("_mean_std")
        ]
        if not metric_columns:
            metric_columns = list(TABLE1_METRIC_COLUMNS)

    meta_cols = [
        c
        for c in ("experiment", "display_name", "robot", "terrain", "workflow", "n_seeds")
        if c in aggregated_df.columns
    ]
    out = aggregated_df[meta_cols].copy() if meta_cols else pd.DataFrame(index=aggregated_df.index)

    for col in metric_columns:
        mean_std_col = f"{col}_mean_std"
        if mean_std_col in aggregated_df.columns:
            out[col] = aggregated_df[mean_std_col]
        elif f"{col}_mean" in aggregated_df.columns:
            out[col] = aggregated_df.apply(
                lambda r, c=col: format_mean_std(r.get(f"{c}_mean"), r.get(f"{c}_std")),
                axis=1,
            )
    # Stable row order matching TABLE1_CONDITIONS
    if "experiment" in out.columns:
        order = {name: i for i, name in enumerate(TABLE1_CONDITIONS.keys())}
        out = out.sort_values("experiment", key=lambda s: s.map(lambda x: order.get(x, 999)))
        out = out.reset_index(drop=True)
    return out


def export_table1_csv(table1_df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write Table I DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table1_df.to_csv(path, index=False)
    return path


def export_table1_latex(
    table1_df: pd.DataFrame,
    path: Optional[Union[str, Path]] = None,
    caption: str = "Multi-seed locomotion results (mean $\\pm$ std over seeds).",
    label: str = "tab:table1",
) -> str:
    """Export Table I as a LaTeX tabular snippet for Overleaf.

    Returns the LaTeX string; optionally writes it to ``path``.
    """
    if table1_df.empty:
        latex = "% No seed-tagged runs available yet.\n"
    else:
        display = table1_df.copy()
        # Prefer human-readable name column
        if "display_name" in display.columns:
            display = display.rename(columns={"display_name": "Condition"})
            drop_cols = [c for c in ("experiment", "robot", "terrain", "workflow") if c in display.columns]
            display = display.drop(columns=drop_cols, errors="ignore")
        col_fmt = "l" + "c" * (len(display.columns) - 1)
        latex = display.to_latex(index=False, escape=False, column_format=col_fmt)
        latex = (
            f"% Auto-generated by rsl_rl_performance.ipynb\n"
            f"\\begin{{table}}[t]\n"
            f"\\centering\n"
            f"{latex}"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\end{{table}}\n"
        )

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex)
    return latex


def aggregate_curves_across_seeds(
    all_metrics: Dict,
    experiment: str,
    metric_name: str = "mean_reward",
    num_points: int = 200,
) -> Optional[pd.DataFrame]:
    """Interpolate per-seed curves and return mean ± std over seeds vs step.

    Useful for convergence plots with error bands. Within-run CI
    (``compute_confidence_intervals``) remains available for single-run curves.
    """
    runs = [
        (run_key, run_data)
        for run_key, run_data in all_metrics.items()
        if run_data.get("experiment") == experiment
    ]
    if not runs:
        return None

    series_list = []
    max_step = 0
    for _run_key, run_data in runs:
        metrics = run_data["metrics"]
        patterns = METRIC_PATTERNS.get(metric_name, [metric_name])
        tag = find_metric_name(metrics, patterns)
        if tag is None or tag not in metrics:
            continue
        df = metrics[tag].sort_values("step")
        if df.empty:
            continue
        max_step = max(max_step, int(df["step"].iloc[-1]))
        series_list.append(df[["step", "value"]].copy())

    if not series_list or max_step <= 0:
        return None

    grid = np.linspace(0, max_step, num_points)
    interpolated = []
    for df in series_list:
        steps = df["step"].values.astype(float)
        values = df["value"].values.astype(float)
        if len(steps) < 2:
            continue
        # Only interpolate within each seed's observed range
        mask = (grid >= steps.min()) & (grid <= steps.max())
        y = np.full_like(grid, np.nan, dtype=float)
        y[mask] = np.interp(grid[mask], steps, values)
        interpolated.append(y)

    if not interpolated:
        return None

    stacked = np.vstack(interpolated)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
    return pd.DataFrame(
        {
            "step": grid,
            "mean": mean,
            "std": std,
            "n_seeds": stacked.shape[0],
            "experiment": experiment,
        }
    )


def plot_seed_mean_std_bars(
    aggregated_df: pd.DataFrame,
    metric_column: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    experiments: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """Bar chart of across-seed mean with seed-std error bars."""
    if aggregated_df.empty:
        print("No aggregated seed data to plot.")
        return ax

    data = aggregated_df.copy()
    if experiments is not None:
        data = data[data["experiment"].isin(experiments)]
    mean_col = f"{metric_column}_mean"
    std_col = f"{metric_column}_std"
    if mean_col not in data.columns:
        print(f"Missing column {mean_col}")
        return ax

    if "experiment" in data.columns:
        order = {name: i for i, name in enumerate(TABLE1_CONDITIONS.keys())}
        data = data.sort_values("experiment", key=lambda s: s.map(lambda x: order.get(x, 999)))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data))
    means = pd.to_numeric(data[mean_col], errors="coerce").fillna(0.0).values
    stds = (
        pd.to_numeric(data[std_col], errors="coerce").fillna(0.0).values
        if std_col in data.columns
        else np.zeros_like(means)
    )
    labels = (
        data["display_name"].tolist()
        if "display_name" in data.columns
        else data["experiment"].tolist()
    )

    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85, edgecolor="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel or metric_column, fontweight="bold")
    ax.set_title(title or f"{metric_column} (mean ± std over seeds)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_seed_curves_with_error_bands(
    all_metrics: Dict,
    experiments: Sequence[str],
    metric_name: str = "mean_reward",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """Plot mean training curves with ±1 seed-std bands for each experiment."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    plotted = False
    for exp_name in experiments:
        curve = aggregate_curves_across_seeds(all_metrics, exp_name, metric_name=metric_name)
        if curve is None or curve.empty:
            continue
        label = TABLE1_CONDITIONS.get(exp_name, {}).get("display_name", exp_name)
        ax.plot(curve["step"], curve["mean"], label=label, linewidth=2)
        ax.fill_between(
            curve["step"],
            curve["mean"] - curve["std"],
            curve["mean"] + curve["std"],
            alpha=0.2,
        )
        plotted = True

    if not plotted:
        print(f"No seed curves available for metric '{metric_name}'")
        return ax

    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel(metric_name, fontweight="bold")
    ax.set_title(title or f"Convergence: {metric_name} (mean ± std over seeds)", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def filter_metrics_by_comparison_group(
    aggregated_df: pd.DataFrame,
    comparison_group: str,
) -> pd.DataFrame:
    """Subset an aggregated seed DataFrame to experiments in a COMPARISONS group."""
    if comparison_group not in COMPARISONS:
        print(f"Unknown comparison group: {comparison_group}")
        return pd.DataFrame()
    exp_names = [exp for exp, _, _, _ in COMPARISONS[comparison_group]]
    if "experiment" not in aggregated_df.columns:
        return pd.DataFrame()
    return aggregated_df[aggregated_df["experiment"].isin(exp_names)].copy()
