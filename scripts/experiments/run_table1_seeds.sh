#!/usr/bin/env bash
# Table I multi-seed training grid (8 conditions × seeds 42, 0, 1 = 24 runs).
# Every condition uses --max_iterations=1500.
#
# Do NOT mix older campaign logs into analysis — only use runs tagged
# *_seed{N} from this campaign (or filter by timestamp after these configs).
#
# Usage (from anywhere):
#   bash scripts/experiments/run_table1_seeds.sh
# Optional single-GPU pin:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_table1_seeds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(42 0 1)
MAX_ITERATIONS=1500

CONDITIONS=(
  "Quadrrl-Velocity-Flat-ANYmal-C-Direct-v0"
  "Quadrrl-Velocity-Rough-ANYmal-C-Direct-v0"
  "Quadrrl-Velocity-Flat-ANYmal-C-v0"
  "Quadrrl-Velocity-Rough-ANYmal-C-v0"
  "Quadrrl-Velocity-Flat-ANYmal-D-v0"
  "Quadrrl-Velocity-Rough-ANYmal-D-v0"
  "Quadrrl-Velocity-Flat-Unitree-Go2-v0"
  "Quadrrl-Velocity-Rough-Unitree-Go2-v0"
  "Quadrrl-Velocity-Flat-Unitree-B2-v0"
  "Quadrrl-Velocity-Rough-Unitree-B2-v0"
)

for task in "${CONDITIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== ${task} seed=${seed} max_iterations=${MAX_ITERATIONS} ==="
    python scripts/reinforcement_learning/rsl_rl/train.py \
      --task="${task}" \
      --num_envs=4096 \
      --seed="${seed}" \
      --run_name="seed${seed}" \
      --max_iterations="${MAX_ITERATIONS}" \
      --headless \
      --logger=tensorboard
  done
done

echo "All 24 Table I seed runs finished."
