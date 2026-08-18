#!/usr/bin/env bash
# Legged vs wheeled-legged multi-seed training grid.
#
# Four morphological pairs (flat + rough), seeds 42 / 0 / 1, 20000 iterations:
#   Unitree Go2  vs Go2W
#   Unitree B2   vs B2W
#   Zsibot ZSL1  vs ZSL1W
#   Deeprobotics Lite3 vs M20
#
# 4 pairs × 2 morphologies × 2 terrains × 3 seeds = 48 runs.
#
# Do NOT mix older campaign logs into analysis — only use runs tagged
# *_seed{N} from this campaign (or filter by timestamp after these configs).
#
# Usage (from anywhere):
#   bash scripts/experiments/run_legged_vs_wheeled_seeds.sh
# Optional single-GPU pin:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_legged_vs_wheeled_seeds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(42 0 1)
MAX_ITERATIONS=20000

# Paired so each morphology is trained on both terrains before the next robot.
CONDITIONS=(
  "Quadrrl-Velocity-Flat-Unitree-Go2-v0"
  "Quadrrl-Velocity-Rough-Unitree-Go2-v0"
  "Quadrrl-Velocity-Flat-Unitree-Go2W-v0"
  "Quadrrl-Velocity-Rough-Unitree-Go2W-v0"
  "Quadrrl-Velocity-Flat-Unitree-B2-v0"
  "Quadrrl-Velocity-Rough-Unitree-B2-v0"
  "Quadrrl-Velocity-Flat-Unitree-B2W-v0"
  "Quadrrl-Velocity-Rough-Unitree-B2W-v0"
  "Quadrrl-Velocity-Flat-Zsibot-ZSL1-v0"
  "Quadrrl-Velocity-Rough-Zsibot-ZSL1-v0"
  "Quadrrl-Velocity-Flat-Zsibot-ZSL1W-v0"
  "Quadrrl-Velocity-Rough-Zsibot-ZSL1W-v0"
  "Quadrrl-Velocity-Flat-Deeprobotics-Lite3-v0"
  "Quadrrl-Velocity-Rough-Deeprobotics-Lite3-v0"
  "Quadrrl-Velocity-Flat-Deeprobotics-M20-v0"
  "Quadrrl-Velocity-Rough-Deeprobotics-M20-v0"
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

echo "All 48 legged vs wheeled seed runs finished."
