# Newton Integration and Sim2Sim Workflow

This guide describes the Newton backend integration implemented in Quadrrl for:

- `Unitree Go2`
- `Unitree B2`
- `Spot`
- `ANYmal-D`

The workflow keeps PhysX task IDs unchanged and adds Newton task variants for side-by-side training and evaluation.

## Prerequisites

- Isaac Lab `develop` branch with Newton integration installed.
- A working Python environment (for example `isaac_lab`) with Quadrrl installed in editable mode.
- NVIDIA GPU and drivers compatible with Isaac Lab requirements.

Reference setup docs:

- <https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/installation.html>
- <https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/solver-transitioning.html>

## Newton Task IDs

The following task IDs are now registered:

- `Quadrrl-Velocity-Flat-Unitree-Go2-Newton-v0`
- `Quadrrl-Velocity-Rough-Unitree-Go2-Newton-v0`
- `Quadrrl-Velocity-Flat-Unitree-B2-Newton-v0`
- `Quadrrl-Velocity-Rough-Unitree-B2-Newton-v0`
- `Quadrrl-Velocity-Flat-Spot-Newton-v0`
- `Quadrrl-Velocity-Rough-Spot-Newton-v0`
- `Quadrrl-Velocity-Flat-Anymal-D-Newton-v0`
- `Quadrrl-Velocity-Rough-Anymal-D-Newton-v0`

## Backend-Safe Physics Notes

- PhysX-only simulation tuning now uses guarded backend-safe setup in:
  - `source/quadrrl/quadrrl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Root property randomization no longer hard-codes `root_physx_view`; it resolves available root physics views:
  - `source/quadrrl/quadrrl/tasks/manager_based/locomotion/velocity/mdp/events.py`
  - `source/quadrrl/quadrrl/tasks/manager_based/locomotion/velocity/backend_utils.py`
- Newton solver budget (`njmax` / `nefc_per_env`) is set from `configure_newton_sim(...)` in
  `source/quadrrl/quadrrl/tasks/manager_based/locomotion/velocity/backend_utils.py`.
  If runtime prints `nefc overflow`, increase this value further.

## Train and Evaluate

### Smoke Check (Zero / Random Agent)

```bash
python scripts/zero_agent.py --task Quadrrl-Velocity-Rough-Unitree-Go2-Newton-v0 --num_envs 32
python scripts/random_agent.py --task Quadrrl-Velocity-Rough-Unitree-Go2-Newton-v0 --num_envs 32
```

### Train (RSL-RL)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Quadrrl-Velocity-Rough-Unitree-Go2-Newton-v0 --num_envs 2048
python scripts/reinforcement_learning/rsl_rl/train.py --task Quadrrl-Velocity-Rough-Unitree-B2-Newton-v0 --num_envs 2048
python scripts/reinforcement_learning/rsl_rl/train.py --task Quadrrl-Velocity-Rough-Spot-Newton-v0 --num_envs 2048
python scripts/reinforcement_learning/rsl_rl/train.py --task Quadrrl-Velocity-Rough-Anymal-D-Newton-v0 --num_envs 2048
```

### Evaluate and Export

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Quadrrl-Velocity-Rough-Unitree-Go2-Newton-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 64
```

The export folder now includes:

- `policy.pt`
- `policy.onnx`
- `deployment_manifest.json`

## Sim2Sim Transfer (PhysX <-> Newton)

New script:

- `scripts/sim2sim_transfer/rsl_rl_transfer.py`

Mapping configs:

- `scripts/sim2sim_transfer/config/physx_to_newton_go2.yaml`
- `scripts/sim2sim_transfer/config/newton_to_physx_go2.yaml`
- `scripts/sim2sim_transfer/config/physx_to_newton_b2.yaml`
- `scripts/sim2sim_transfer/config/newton_to_physx_b2.yaml`
- `scripts/sim2sim_transfer/config/physx_to_newton_spot.yaml`
- `scripts/sim2sim_transfer/config/newton_to_physx_spot.yaml`
- `scripts/sim2sim_transfer/config/physx_to_newton_anymal_d.yaml`
- `scripts/sim2sim_transfer/config/newton_to_physx_anymal_d.yaml`

### PhysX -> Newton Example

```bash
python scripts/sim2sim_transfer/rsl_rl_transfer.py \
  --task Quadrrl-Velocity-Rough-Unitree-Go2-Newton-v0 \
  --checkpoint /path/to/physx_checkpoint.pt \
  --policy_transfer_file scripts/sim2sim_transfer/config/physx_to_newton_go2.yaml \
  --num_envs 32
```

## Validation Matrix (This Implementation)

Executed checks in this workspace:

- `python -m compileall ...` on modified velocity configs, transfer script, and play script: passed.
- YAML schema validation for all `scripts/sim2sim_transfer/config/*.yaml`: passed.
- Runtime smoke commands (train/transfer) were attempted in `isaac_lab` conda env and reached environment import.

Known runtime blocker observed in this workspace:

- `ImportError: cannot import name 'AdditiveUniformNoiseCfg' from isaaclab.utils.noise`
  - Triggered while importing `velocity_env_cfg.py`.
  - This indicates an Isaac Lab API mismatch between this Quadrrl code and the local Isaac Lab package version in the active environment.

### Newton -> PhysX Example

```bash
python scripts/sim2sim_transfer/rsl_rl_transfer.py \
  --task Quadrrl-Velocity-Rough-Unitree-Go2-v0 \
  --checkpoint /path/to/newton_checkpoint.pt \
  --policy_transfer_file scripts/sim2sim_transfer/config/newton_to_physx_go2.yaml \
  --num_envs 32
```

## Deployment Contract (Sim-to-Real Preparation)

For each evaluated policy, treat these as deployment artifacts:

- `policy.pt` (TorchScript)
- `policy.onnx` (interop export)
- `deployment_manifest.json`:
  - task and checkpoint provenance
  - control step (`step_dt`)
  - action/observation space descriptors
  - deployment notes

To keep deployment consistent across sim and hardware:

1. Preserve observation ordering expected by the policy.
2. Preserve action ordering and scaling.
3. Match controller frequency to exported policy assumptions (`step_dt` / decimation).
4. Keep joint naming and ordering aligned with the selected transfer YAML.
