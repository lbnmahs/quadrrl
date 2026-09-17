#!/usr/bin/env python3
# Copyright (c) 2024-2025, Laban Njoroge Mahihu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test / headless driver for legged-vs-wheeled performance analysis.

Mirrors ``notebooks/legged_vs_wheeled_performance.ipynb`` without requiring
Jupyter: loads 48 seed runs, writes tables/figures, and spot-checks Go2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import rsl_rl_analysis_utils as utils  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "rsl_rl",
        help="RSL-RL logs directory",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "exports" / "legged_vs_wheeled",
        help="Export directory for CSV/LaTeX/PNG",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    export_dir = utils.resolve_legged_vs_wheeled_export_dir(args.export_dir)
    fig_dir = export_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = (42, 0, 1)
    experiments = list(utils.LEGGED_VS_WHEELED_CONDITIONS.keys())
    max_iters = utils.LEGGED_VS_WHEELED_MAX_ITERATIONS
    expected = len(experiments) * len(seeds)

    print(f"Loading seed-tagged metrics from {logs_dir} …")
    all_metrics = utils.load_seed_tagged_metrics(
        logs_dir,
        experiments=experiments,
        seeds=seeds,
        conditions=utils.LEGGED_VS_WHEELED_CONDITIONS,
    )
    print(f"Loaded {len(all_metrics)} / {expected} runs")
    if len(all_metrics) != expected:
        missing = []
        have = {(rd["experiment"], rd.get("seed")) for rd in all_metrics.values()}
        for exp in experiments:
            for seed in seeds:
                if (exp, seed) not in have:
                    missing.append(f"{exp} seed={seed}")
        raise SystemExit("Missing runs:\n  " + "\n  ".join(missing))

    per_seed = utils.extract_per_seed_terminal_metrics(
        all_metrics,
        metric_keys=utils.LEGGED_VS_WHEELED_METRIC_COLUMNS,
        include_convergence=False,
    )
    aggregated = utils.aggregate_across_seeds(
        per_seed, metric_columns=utils.LEGGED_VS_WHEELED_METRIC_COLUMNS
    )
    utils.export_morphology_terminal_tables(aggregated, export_dir=export_dir)

    paired_delta = utils.build_morphology_paired_delta_dataframe(per_seed)
    utils.export_morphology_csv(paired_delta, "paired_deltas.csv", export_dir=export_dir)

    snapshot_agg = utils.snapshot_at_fractions(
        all_metrics,
        metric_key="mean_reward",
        fractions=utils.LEGGED_VS_WHEELED_STAGE_FRACTIONS,
        max_iterations=max_iters,
        per_seed=False,
    )
    snapshot_per_seed = utils.snapshot_at_fractions(
        all_metrics,
        metric_key="mean_reward",
        fractions=utils.LEGGED_VS_WHEELED_STAGE_FRACTIONS,
        max_iterations=max_iters,
        per_seed=True,
    )
    delta_per_seed = utils.stage_deltas_from_snapshots(snapshot_per_seed)
    delta_seed_agg = utils.aggregate_across_seeds(
        delta_per_seed,
        metric_columns=["delta"],
        group_cols=[
            c
            for c in (
                "experiment",
                "display_name",
                "category",
                "robot",
                "terrain",
                "pair_id",
                "metric_key",
                "window",
                "fraction_lo",
                "fraction_hi",
            )
            if c in delta_per_seed.columns
        ],
    )
    delta_for_export = delta_seed_agg.rename(
        columns={
            "delta_mean": "mean",
            "delta_std": "std",
            "delta_mean_std": "mean_std",
        }
    )
    utils.export_morphology_stage_tables(
        snapshot_agg, delta_df=delta_for_export, export_dir=export_dir
    )

    # Figures
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, metric in zip(
        axes.ravel(),
        ["mean_reward", "track_lin_vel", "track_ang_vel", "episode_length"],
    ):
        utils.plot_seed_mean_std_bars(
            aggregated,
            metric_column=metric,
            title=metric,
            experiments=experiments,
            ax=ax,
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "flat_vs_rough_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_df = aggregated[
        ["pair_id", "terrain", "category", "mean_reward_mean", "mean_reward_std"]
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, terrain in zip(axes, ("flat", "rough")):
        sub = plot_df[plot_df["terrain"] == terrain]
        x = np.arange(len(utils.MORPHOLOGY_PAIRS))
        width = 0.35
        for offset, category, color in (
            (-width / 2, "legged", "#4C72B0"),
            (width / 2, "wheeled", "#DD8452"),
        ):
            means, stds = [], []
            for pair_id in utils.MORPHOLOGY_PAIRS:
                r = sub[(sub["pair_id"] == pair_id) & (sub["category"] == category)]
                means.append(float(r["mean_reward_mean"].iloc[0]) if len(r) else 0.0)
                stds.append(float(r["mean_reward_std"].iloc[0]) if len(r) else 0.0)
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=category,
                color=color,
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(utils.MORPHOLOGY_PAIRS.keys()))
        ax.set_title(f"Paired morphology — {terrain}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "paired_morphology_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    heat = delta_for_export.pivot_table(
        index="experiment", columns="window", values="mean", aggfunc="first"
    ).reindex(experiments)
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Stage ΔR heatmap")
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    windows = list(delta_for_export["window"].drop_duplicates())
    x = np.arange(len(experiments))
    width = 0.8 / max(len(windows), 1)
    palette = sns.color_palette("viridis", n_colors=max(len(windows), 1))
    for i, window in enumerate(windows):
        means, stds = [], []
        for exp in experiments:
            row = delta_for_export[
                (delta_for_export["experiment"] == exp)
                & (delta_for_export["window"] == window)
            ]
            means.append(
                float(row["mean"].iloc[0])
                if len(row) and pd.notna(row["mean"].iloc[0])
                else 0.0
            )
            stds.append(
                float(row["std"].iloc[0])
                if len(row) and "std" in row.columns and pd.notna(row["std"].iloc[0])
                else 0.0
            )
        ax.bar(
            x + (i - (len(windows) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=window,
            color=palette[i],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [utils.LEGGED_VS_WHEELED_CONDITIONS[e]["display_name"] for e in experiments],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Δ mean_reward")
    ax.set_title("Stage learning rates")
    ax.legend(title="window")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_delta_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for pair_id, (legged_stem, wheeled_stem) in utils.MORPHOLOGY_PAIRS.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, terrain in zip(axes, ("flat", "rough")):
            exps = [f"{legged_stem}_{terrain}", f"{wheeled_stem}_{terrain}"]
            utils.plot_seed_curves_with_error_bands(
                all_metrics,
                experiments=exps,
                metric_name="mean_reward",
                title=f"{pair_id} — {terrain}",
                ax=ax,
            )
            for step in (5000, 10000, 15000, 20000):
                ax.axvline(step, color="gray", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xlim(0, max_iters)
        fig.tight_layout()
        fig.savefig(fig_dir / f"convergence_{pair_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Spot-check Go2 vs Go2W
    check_exps = [
        "unitree_go2_flat",
        "unitree_go2_rough",
        "unitree_go2w_flat",
        "unitree_go2w_rough",
    ]
    term = aggregated[aggregated["experiment"].isin(check_exps)][
        ["experiment", "mean_reward_mean"]
    ].set_index("experiment")
    r100 = (
        snapshot_agg[snapshot_agg["fraction"] == 1.0][["experiment", "mean"]]
        .set_index("experiment")
        .rename(columns={"mean": "R_100"})
    )
    r25 = (
        snapshot_agg[snapshot_agg["fraction"] == 0.25][["experiment", "mean"]]
        .set_index("experiment")
        .rename(columns={"mean": "R_25"})
    )
    spot = term.join(r100).join(r25)
    spot["abs_term_R100"] = (spot["mean_reward_mean"] - spot["R_100"]).abs()
    print(spot.to_string())
    if not (spot["abs_term_R100"] < 1e-2).all():
        raise SystemExit("Spot-check failed: R_100 does not match terminal mean_reward")
    improving = (spot["R_100"] > spot["R_25"] + 1e-3).sum()
    print(f"Improving Go2/Go2W conditions (R_100 > R_25): {improving} / {len(spot)}")

    required = [
        export_dir / "terminal_mean_std.csv",
        export_dir / "terminal_mean_std.tex",
        export_dir / "stage_snapshots.csv",
        export_dir / "stage_snapshots.tex",
        export_dir / "stage_deltas.csv",
        fig_dir / "paired_morphology_bars.png",
        fig_dir / "stage_delta_bars.png",
    ]
    missing_files = [str(p) for p in required if not p.exists()]
    if missing_files:
        raise SystemExit("Missing exports:\n  " + "\n  ".join(missing_files))

    print("OK — exports under", export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
