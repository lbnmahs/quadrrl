#!/usr/bin/env python3
# Copyright (c) 2024-2025, Laban Njoroge Mahihu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to analyze training logs and extract key metrics.
This script scans the logs directory and provides insights into:
- Log structure and organization
- Key metrics tracked during training
- Training progress and statistics
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import yaml

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def scan_log_structure(logs_dir: Path) -> Dict[str, Any]:
    """Scan the logs directory and return its structure."""
    structure = {
        "frameworks": {},
        "total_runs": 0,
        "total_checkpoints": 0,
        "total_tensorboard_logs": 0,
    }

    for framework_dir in logs_dir.iterdir():
        if not framework_dir.is_dir():
            continue

        framework_name = framework_dir.name
        structure["frameworks"][framework_name] = {
            "experiments": {},
            "total_runs": 0,
        }

        # Scan experiments
        for exp_dir in framework_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name
            runs = []

            # Scan runs
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                run_info = {
                    "path": str(run_dir),
                    "name": run_dir.name,
                    "checkpoints": [],
                    "tensorboard_logs": [],
                    "has_params": False,
                    "has_exported": False,
                    "has_videos": False,
                }

                # Check for model checkpoints
                for item in run_dir.iterdir():
                    if item.is_file():
                        if item.suffix == ".pt" and "model" in item.name:
                            run_info["checkpoints"].append(item.name)
                            structure["total_checkpoints"] += 1
                        elif item.name.startswith("events.out.tfevents"):
                            run_info["tensorboard_logs"].append(item.name)
                            structure["total_tensorboard_logs"] += 1
                    elif item.is_dir():
                        if item.name == "params":
                            run_info["has_params"] = True
                            # Try to read config files
                            env_yaml = item / "env.yaml"
                            agent_yaml = item / "agent.yaml"
                            if env_yaml.exists():
                                try:
                                    with open(env_yaml, 'r') as f:
                                        run_info["env_config"] = yaml.safe_load(f)
                                except Exception:
                                    pass
                            if agent_yaml.exists():
                                try:
                                    with open(agent_yaml, 'r') as f:
                                        run_info["agent_config"] = yaml.safe_load(f)
                                except Exception:
                                    pass
                        elif item.name == "exported":
                            run_info["has_exported"] = True
                        elif item.name == "videos":
                            run_info["has_videos"] = True

                if run_info["checkpoints"] or run_info["tensorboard_logs"]:
                    runs.append(run_info)
                    structure["total_runs"] += 1
                    structure["frameworks"][framework_name]["total_runs"] += 1

            if runs:
                structure["frameworks"][framework_name]["experiments"][exp_name] = runs

    return structure


def extract_tensorboard_metrics(event_file: Path) -> Dict[str, List[Any]]:
    """Extract metrics from a TensorBoard event file."""
    if not TENSORBOARD_AVAILABLE:
        return {}

    try:
        ea = event_accumulator.EventAccumulator(
            str(event_file.parent),
            size_guidance={
                event_accumulator.SCALARS: 0,  # 0 = load all
            }
        )
        ea.Reload()

        metrics = {}
        scalar_tags = ea.Tags()['scalars']

        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            metrics[tag] = [
                {
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                }
                for event in scalar_events
            ]

        return metrics
    except Exception as e:
        return {"error": str(e)}


def analyze_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Analyze extracted metrics and return summary statistics."""
    analysis = {
        "metric_names": list(metrics.keys()),
        "summary": {},
    }

    for metric_name, values in metrics.items():
        if not values or isinstance(values, dict):
            continue

        metric_values = [v["value"] for v in values]
        analysis["summary"][metric_name] = {
            "count": len(metric_values),
            "min": min(metric_values) if metric_values else None,
            "max": max(metric_values) if metric_values else None,
            "mean": sum(metric_values) / len(metric_values) if metric_values else None,
            "last_value": metric_values[-1] if metric_values else None,
            "first_value": metric_values[0] if metric_values else None,
        }

    return analysis


def identify_key_metrics(all_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """Identify the most significant metrics based on common naming patterns."""
    key_patterns = [
        "reward", "return", "episode",
        "loss", "value", "policy",
        "distance", "position", "tracking",
        "success", "completion", "termination",
        "fps", "time", "steps",
    ]

    key_metrics = []
    metric_scores = defaultdict(int)

    for run_name, metrics in all_metrics.items():
        for metric_name in metrics.get("metric_names", []):
            metric_lower = metric_name.lower()
            for pattern in key_patterns:
                if pattern in metric_lower:
                    metric_scores[metric_name] += 1
                    break

    # Sort by frequency and return top metrics
    sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
    key_metrics = [name for name, _ in sorted_metrics[:20]]

    return key_metrics


def generate_report(logs_dir: Path, sample_runs: int = 5) -> Dict[str, Any]:
    """Generate a comprehensive report of the logs."""
    print(f"Scanning logs directory: {logs_dir}")
    structure = scan_log_structure(logs_dir)

    print(f"\nFound {structure['total_runs']} training runs across {len(structure['frameworks'])} frameworks")

    # Collect metrics from sample runs
    all_metrics = {}
    sample_count = 0

    for framework_name, framework_data in structure["frameworks"].items():
        print(f"\nFramework: {framework_name} ({framework_data['total_runs']} runs)")
        for exp_name, runs in framework_data["experiments"].items():
            print(f"  Experiment: {exp_name} ({len(runs)} runs)")

            # Sample runs for metric extraction
            for run in runs[: min(sample_runs, len(runs))]:
                if sample_count >= sample_runs:
                    break

                if run["tensorboard_logs"]:
                    event_file = Path(run["path"]) / run["tensorboard_logs"][0]
                    print(f"    Analyzing: {run['name']}")
                    metrics = extract_tensorboard_metrics(event_file)
                    if metrics and "error" not in metrics:
                        all_metrics[f"{framework_name}/{exp_name}/{run['name']}"] = analyze_metrics(metrics)
                        sample_count += 1

    # Identify key metrics
    key_metrics = identify_key_metrics(all_metrics)

    report = {
        "structure": structure,
        "sample_metrics": all_metrics,
        "key_metrics": key_metrics,
    }

    return report


def print_summary(report: Dict[str, Any]):
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 80)
    print("LOGS ANALYSIS SUMMARY")
    print("=" * 80)

    structure = report["structure"]

    print("\n📊 OVERVIEW:")
    print(f"  Total training runs: {structure['total_runs']}")
    print(f"  Total model checkpoints: {structure['total_checkpoints']}")
    print(f"  Total TensorBoard logs: {structure['total_tensorboard_logs']}")

    print("\n📁 FRAMEWORKS:")
    for framework_name, framework_data in structure["frameworks"].items():
        print(f"  • {framework_name}: {framework_data['total_runs']} runs")
        for exp_name, runs in framework_data["experiments"].items():
            print(f"    - {exp_name}: {len(runs)} runs")
            # Show sample run info
            for run in runs[:2]:
                checkpoint_count = len(run["checkpoints"])
                print(f"      • {run['name']}: {checkpoint_count} checkpoints")

    if report["key_metrics"]:
        print("\n🎯 KEY METRICS IDENTIFIED:")
        for i, metric in enumerate(report["key_metrics"][:15], 1):
            print(f"  {i}. {metric}")

    # Print sample metric summaries
    if report["sample_metrics"]:
        print("\n📈 SAMPLE METRIC SUMMARIES:")
        for run_name, metrics_data in list(report["sample_metrics"].items())[:3]:
            print(f"\n  Run: {run_name}")
            summary = metrics_data.get("summary", {})

            # Show a few key metrics
            for metric_name in list(summary.keys())[:5]:
                stats = summary[metric_name]
                if stats["count"] > 0:
                    print(f"    • {metric_name}:")
                    print(f"      - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    print(f"      - Mean: {stats['mean']:.4f}")
                    print(f"      - Last: {stats['last_value']:.4f}")
                    print(f"      - Samples: {stats['count']}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Path to logs directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--sample-runs",
        type=int,
        default=5,
        help="Number of sample runs to analyze in detail",
    )

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return

    # Generate report
    report = generate_report(logs_dir, sample_runs=args.sample_runs)

    # Print summary
    print_summary(report)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()
