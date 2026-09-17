#!/usr/bin/env bash
# Fair-Morph-v2 multi-seed training grid (comparable legged vs wheeled).
#
# Go2↔Go2W, B2↔B2W, ZSL1↔ZSL1W, Lite3↔M20 (flat + rough), seeds 42 / 0 / 1 / 2 / 3, 20000 iters:
#   Matched tracking rewards (1.5 / 0.75), upward=0 on wheeled, equal MLP capacity.
#
# 4 pairs × 2 morphologies × 2 terrains × 5 seeds = 80 runs.
#
# Re-running this script skips seeds that already have model_19999.pt and
# continues after a failed seed instead of aborting the remaining grid.
#
# Native *-v0 / finished 48-run campaign logs are untouched (Fair uses *_fair dirs).
#
# Usage (from anywhere):
#   bash scripts/experiments/run_fair_morph_v2_seeds.sh
# Optional single-GPU pin:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_fair_morph_v2_seeds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(42 0 1 2 3)
MAX_ITERATIONS=20000

CONDITIONS=(
  "Quadrrl-Velocity-Flat-Unitree-Go2-Fair-v0"
  "Quadrrl-Velocity-Rough-Unitree-Go2-Fair-v0"
  "Quadrrl-Velocity-Flat-Unitree-Go2W-Fair-v0"
  "Quadrrl-Velocity-Rough-Unitree-Go2W-Fair-v0"
  "Quadrrl-Velocity-Flat-Unitree-B2-Fair-v0"
  "Quadrrl-Velocity-Rough-Unitree-B2-Fair-v0"
  "Quadrrl-Velocity-Flat-Unitree-B2W-Fair-v0"
  "Quadrrl-Velocity-Rough-Unitree-B2W-Fair-v0"
  "Quadrrl-Velocity-Flat-Zsibot-ZSL1-Fair-v0"
  "Quadrrl-Velocity-Rough-Zsibot-ZSL1-Fair-v0"
  "Quadrrl-Velocity-Flat-Zsibot-ZSL1W-Fair-v0"
  "Quadrrl-Velocity-Rough-Zsibot-ZSL1W-Fair-v0"
  "Quadrrl-Velocity-Flat-Deeprobotics-Lite3-Fair-v0"
  "Quadrrl-Velocity-Rough-Deeprobotics-Lite3-Fair-v0"
  "Quadrrl-Velocity-Flat-Deeprobotics-M20-Fair-v0"
  "Quadrrl-Velocity-Rough-Deeprobotics-M20-Fair-v0"
)

# Quadrrl-Velocity-{Flat|Rough}-{Robot}-Fair-v0 -> {robot}_{flat|rough}_fair
task_to_experiment() {
  local rest="${1#Quadrrl-Velocity-}"
  rest="${rest%-v0}"
  rest="${rest%-Fair}"
  local terrain="${rest%%-*}"
  local robot="${rest#*-}"
  robot="$(echo "${robot}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
  terrain="$(echo "${terrain}" | tr '[:upper:]' '[:lower:]')"
  echo "${robot}_${terrain}_fair"
}

run_is_complete() {
  local exp_name="$1"
  local seed="$2"
  local final_iter=$((MAX_ITERATIONS - 1))
  local run_dir
  run_dir="$(ls -1d logs/rsl_rl/"${exp_name}"/*_seed"${seed}" 2>/dev/null | tail -1 || true)"
  [[ -n "${run_dir}" && -f "${run_dir}/model_${final_iter}.pt" ]]
}

FAILED_RUNS=()
SKIPPED_RUNS=0

for task in "${CONDITIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    exp_name="$(task_to_experiment "${task}")"
    if run_is_complete "${exp_name}" "${seed}"; then
      echo "=== SKIP ${task} seed=${seed} (already has model_$((MAX_ITERATIONS - 1)).pt) ==="
      SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
      continue
    fi
    echo "=== ${task} seed=${seed} max_iterations=${MAX_ITERATIONS} exp=${exp_name} ==="
    if python scripts/reinforcement_learning/rsl_rl/train.py \
      --task="${task}" \
      --num_envs=1024 \
      --seed="${seed}" \
      --run_name="seed${seed}" \
      --max_iterations="${MAX_ITERATIONS}" \
      --headless \
      --logger=tensorboard; then
      :
    else
      echo "FAILED: ${task} seed=${seed}"
      FAILED_RUNS+=("${task} seed=${seed}")
    fi
  done
done

echo "Skipped ${SKIPPED_RUNS} already-complete seed runs."
if ((${#FAILED_RUNS[@]})); then
  echo "Failed ${#FAILED_RUNS[@]} run(s):"
  printf '  %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi
echo "All 80 Fair-Morph-v2 seed runs finished."
