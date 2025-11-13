# QUADRRL

## Overview

Quadrrl builds on NVIDIA Isaac Lab to research and prototype deep reinforcement learning for quadruped robots.
It includes locomotion and navigation tasks across flat and rough terrains, covering velocity tracking,
goal-directed navigation, and multi-agent coordination.

**Highlights**
- Unified training suite for ANYmal-C, ANYmal-D, and Unitree Go2 quadrupeds
- Direct and manager-based task variants for locomotion and navigation
- Ready-to-run configs for `rl_games`, `rsl_rl`, and `skrl` RL(Reinforcement Learning) frameworks
- Optional Omniverse UI extension for quick visualization and debugging

## 📋 Prerequisites

### Compute Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **CPU**: Multi-core processor (8+ cores recommended for large batches)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 20 GB+ free space for Isaac Lab and generated logs

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11 (64-bit)
- **Isaac Lab**: Installed per the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python**: 3.10 or newer (conda, uv, or virtualenv)
- **CUDA**: Match the version required by your Isaac Lab build
- **Git**: For cloning this repository
- Optional: Omniverse Kit / Isaac Sim for UI workflows

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/lbnmahs/quadrrl.git
cd quadrrl

# Activate the Python environment that already has Isaac Lab dependencies
conda activate isaaclab    # or `source isaaclab.sh -p` depending on your setup

# Install Quadrrl in editable mode
python -m pip install -e source/quadrrl

# (Optional) install developer tooling
pip install pre-commit
pre-commit install
```

## 🚀 Running Quadrrl

### Discover Tasks

```bash
python scripts/list_envs.py
```

### Train Policies

Replace `<RL_LIBRARY>` with `rl_games`, `rsl_rl`, or `skrl`, and supply any extra training flags.

```bash
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME> --num_envs=4096 --seed=42
```

### Evaluate Saved Policies

Play-mode tasks (suffix `-Play`) load evaluation checkpoints and curriculum settings.

```bash
python scripts/<RL_LIBRARY>/play.py --task=<TASK_NAME>-Play --checkpoint=/absolute/path/to/checkpoint.pth
```

Tip: Use `isaaclab.sh -p` or `isaaclab.bat -p` in place of `python` if Isaac Lab is not installed in the active Python environment.

## 🌍 Available Environments

| S. No. | Task Name                                           | Entry Point                                                         | Config                                                                                                  |
|-------:|-----------------------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 1      | Template-Quadrrl-Velocity-Flat-Anymal-C-Direct-v0   | `quadrrl.tasks.direct.anymal_c.anymal_c_env:AnymalCEnv`             | `quadrrl.tasks.direct.anymal_c.anymal_c_env_cfg:AnymalCFlatEnvCfg`                                      |
| 2      | Template-Quadrrl-Velocity-Rough-Anymal-C-Direct-v0  | `quadrrl.tasks.direct.anymal_c.anymal_c_env:AnymalCEnv`             | `quadrrl.tasks.direct.anymal_c.anymal_c_env_cfg:AnymalCRoughEnvCfg`                                     |
| 3      | Template-Quadrrl-Marl-Direct-v0                     | `quadrrl.tasks.direct.quadrrl_marl.quadrrl_marl_env:QuadrrlMarlEnv` | `quadrrl.tasks.direct.quadrrl_marl.quadrrl_marl_env_cfg:QuadrrlMarlEnvCfg`                              |
| 4      | Template-Quadrrl-Velocity-Flat-Anymal-C-v0          | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anym- `source/quadrrl/pyproject.toml` – package metadata for editable installs.
al_c.flat_env_cfg:AnymalCFlatEnvCfg`        |
| 5      | Template-Quadrrl-Velocity-Flat-Anymal-C-Play-v0     | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg:AnymalCFlatEnvCfg_PLAY`   |
| 6      | Template-Quadrrl-Velocity-Rough-Anymal-C-v0         | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg:AnymalCRoughEnvCfg`      |
| 7      | Template-Quadrrl-Velocity-Rough-Anymal-C-Play-v0    | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg:AnymalCRoughEnvCfg_PLAY` |
| 8      | Template-Quadrrl-Velocity-Flat-Anymal-D-v0          | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg:AnymalDFlatEnvCfg`        |
| 9      | Template-Quadrrl-Velocity-Flat-Anymal-D-Play-v0     | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg:AnymalDFlatEnvCfg_PLAY`   |
| 10     | Template-Quadrrl-Velocity-Rough-Anymal-D-v0         | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_d.rough_env_cfg:AnymalDRoughEnvCfg`      |
| 11     | Template-Quadrrl-Velocity-Rough-Anymal-D-Play-v0    | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_d.rough_env_cfg:AnymalDRoughEnvCfg_PLAY` |
| 12     | Template-Quadrrl-Velocity-Flat-Unitree-Go2-v0       | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg:UnitreeGo2FlatEnvCfg`          |
| 13     | Template-Quadrrl-Velocity-Flat-Unitree-Go2-Play-v0  | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY`     |
| 14     | Template-Quadrrl-Velocity-Rough-Unitree-Go2-v0      | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg:UnitreeGo2RoughEnvCfg`        |
| 15     | Template-Quadrrl-Velocity-Rough-Unitree-Go2-Play-v0 | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY`   |
| 16     | Template-Quadrrl-Navigation-Flat-Anymal-C-v0        | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg:NavigationEnvCfg`            |
| 17     | Template-Quadrrl-Navigation-Flat-Anymal-C-Play-v0   | `isaaclab.envs:ManagerBasedRLEnv`                                   | `quadrrl.tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg:NavigationEnvCfg_PLAY`       |

Update `/home/mahs/Development/quadrrl/scripts/list_envs.py` if you rename any tasks so that they continue to show up in listings.

## 📂 Project Layout

- `source/quadrrl/quadrrl/robots` – robot asset wrappers (ANYmal variants and Unitree Go2).
- `source/quadrrl/quadrrl/tasks/direct` – low-level Isaac Gym–style environments and multi-agent setups with RL configs.
- `source/quadrrl/quadrrl/tasks/manager_based` – manager-based tasks with locomotion and navigation curricula, rewards, and symmetry helpers.
- `source/quadrrl/scripts` – entry points for training, evaluation, and diagnostic agents for supported RL frameworks.

Refer to `scripts/quadrupeds.py` for additional guidance on composing task configurations programmatically.

## 🤝 Contributing

- Fork the repository, create feature branches, and open pull requests with clear descriptions.
- Run `pre-commit run --all-files` before submitting changes.
- Add tests or evaluation scripts when introducing new environments or reward structures.
- Update this README and `docs/CHANGELOG.rst` when you add new tasks or major capabilities.

## 📚 Resources & Inspiration

- NVIDIA Isaac Lab Documentation – <https://isaac-sim.github.io/IsaacLab/>
- Quadrrl training examples in `scripts/quadrupeds.py`

