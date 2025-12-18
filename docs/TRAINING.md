# Training Guide

Complete guide for training single-agent and multi-agent reinforcement learning policies with Quadrrl.

## Available Environments

### Single-Agent Locomotion Tasks
> **Analysis scope:** RSL-RL benchmark analysis focuses on **Unitree Go2**, **ANYmal-C**, and **ANYmal-D** for single-agent velocity-tracking. Spot single-agent runs remain available but are not part of the core RSL-RL comparison; Spot is primarily used for single-vs-multi-agent comparisons with HARL.

#### Direct Single-Agent Locomotion

| S. No. | Task Name                                           | Entry Point          | Config                    |
|-------:|-----------------------------------------------------|----------------------|---------------------------|
| 1      | Template-Quadrrl-Velocity-Flat-Anymal-C-Direct-v0   | `anymal_c_env.py`    | `anymal_c_env_cfg.py`     |
| 2      | Template-Quadrrl-Velocity-Rough-Anymal-C-Direct-v0  | `anymal_c_env.py`    | `anymal_c_env_cfg.py`     |

#### Manager-Based Single-Agent Locomotion

| S. No. | Task Name                                           | Entry Point          | Config                    |
|-------:|-----------------------------------------------------|----------------------|---------------------------|
| 3      | Template-Quadrrl-Velocity-Flat-Anymal-C-v0          | `ManagerBasedRLEnv`  | `flat_env_cfg.py`         |
| 4      | Template-Quadrrl-Velocity-Flat-Anymal-C-Play-v0     | `ManagerBasedRLEnv`  | `flat_env_cfg.py`         |
| 5      | Template-Quadrrl-Velocity-Rough-Anymal-C-v0         | `ManagerBasedRLEnv`  | `rough_env_cfg.py`        |
| 6      | Template-Quadrrl-Velocity-Rough-Anymal-C-Play-v0     | `ManagerBasedRLEnv`  | `rough_env_cfg.py`        |
| 7      | Template-Quadrrl-Velocity-Flat-Anymal-D-v0          | `ManagerBasedRLEnv`  | `flat_env_cfg.py`         |
| 8      | Template-Quadrrl-Velocity-Flat-Anymal-D-Play-v0      | `ManagerBasedRLEnv`  | `flat_env_cfg.py`         |
| 9      | Template-Quadrrl-Velocity-Rough-Anymal-D-v0         | `ManagerBasedRLEnv`  | `rough_env_cfg.py`        |
| 10     | Template-Quadrrl-Velocity-Rough-Anymal-D-Play-v0    | `ManagerBasedRLEnv`  | `rough_env_cfg.py`        |
| 11     | Template-Quadrrl-Velocity-Flat-Unitree-Go2-v0       | `ManagerBasedRLEnv`  | `flat_env_cfg.py`         |
| 12     | Template-Quadrrl-Velocity-Flat-Unitree-Go2-Play-v0  | `ManagerBasedRLEnv`  | `flat_env_cfg.py`         |
| 13     | Template-Quadrrl-Velocity-Rough-Unitree-Go2-v0      | `ManagerBasedRLEnv`  | `rough_env_cfg.py`        |
| 14     | Template-Quadrrl-Velocity-Rough-Unitree-Go2-Play-v0 | `ManagerBasedRLEnv`  | `rough_env_cfg.py`        |
| 15     | Template-Quadrrl-Velocity-Flat-Spot-v0             | `ManagerBasedRLEnv`  | `spot/flat_env_cfg.py`    |
| 16     | Template-Quadrrl-Velocity-Flat-Spot-Play-v0         | `ManagerBasedRLEnv`  | `spot/flat_env_cfg.py`    |
| 17     | Template-Quadrrl-Velocity-Rough-Spot-v0            | `ManagerBasedRLEnv`  | `spot/rough_env_cfg.py`   |
| 18     | Template-Quadrrl-Velocity-Rough-Spot-Play-v0       | `ManagerBasedRLEnv`  | `spot/rough_env_cfg.py`   |

**Spot reward structure note:** Spot uses gait- and contact-focused rewards (gait phase shaping, foot-clearance, air-time balance) that differ from the generic locomotion reward set used by ANYmal/Go2, enabling richer gait coordination studies.

### Single-Agent Navigation Tasks

#### Manager-Based Single-Agent Navigation

| S. No. | Task Name                                           | Entry Point          | Config                    |
|-------:|-----------------------------------------------------|----------------------|---------------------------|
| 19     | Template-Quadrrl-Navigation-Flat-Anymal-C-v0        | `ManagerBasedRLEnv`  | `navigation_env_cfg.py`    |
| 20     | Template-Quadrrl-Navigation-Flat-Anymal-C-Play-v0   | `ManagerBasedRLEnv`  | `navigation_env_cfg.py`    |
| 21     | Template-Quadrrl-Navigation-Rough-Anymal-C-v0      | `ManagerBasedRLEnv`  | `navigation_env_cfg.py`    |
| 22     | Template-Quadrrl-Navigation-Rough-Anymal-C-Play-v0 | `ManagerBasedRLEnv`  | `navigation_env_cfg.py`    |

### Multi-Agent Tasks

#### Direct Multi-Agent

| S. No. | Task Name                                           | Entry Point              | Config                      | RL Framework           |
|-------:|-----------------------------------------------------|--------------------------|-----------------------------|------------------------|
| 23     | Template-Quadrrl-MARL-Direct-Anymal-C-v0           | `anymal_c_marl_env.py`   | `anymal_c_marl_env_cfg.py`   | HARL                   |

#### Manager-Based Multi-Agent Locomotion

| S. No. | Task Name                                           | Entry Point            | Config                      | RL Framework           |
|-------:|-----------------------------------------------------|------------------------|-----------------------------|------------------------|
| 24     | Template-Quadrrl-Velocity-Flat-Spot-MARL-v0        | `ManagerBasedMARLEnv`  | `spot_marl_env_cfg.py`      | HARL (primary), RSL-RL |
| 25     | Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0   | `ManagerBasedMARLEnv`  | `spot_marl_env_cfg.py`      | HARL (primary), RSL-RL |

> **Comparison plan:** Spot is the primary benchmark for single-agent vs multi-agent reinforcement learning using HARL (leg-level agents). RSL-RL analysis remains focused on Unitree Go2, ANYmal-C, and ANYmal-D for single-agent baselines.

**Note:** 
- Update `scripts/list_envs.py` if you rename any tasks so that they continue to show up in listings.
- **MARL tasks are not fully fine-tuned and are still being worked on.**

## Single-Agent Reinforcement Learning

### Training

Replace `<RL_LIBRARY>` with `rl_games`, `rsl_rl`, `skrl`, or `harl`, and supply any extra training flags.

**Linux:**
```bash
python scripts/reinforcement_learning/<RL_LIBRARY>/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --seed=42
```

**Windows:**
```cmd
isaaclab.bat -p scripts/reinforcement_learning/<RL_LIBRARY>/train.py ^
    --task=<TASK_NAME> ^
    --num_envs=4096 ^
    --seed=42
```

**Example:**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-v0 \
    --num_envs=4096 \
    --seed=42
```

### Evaluation

Add the suffix `-Play` to load evaluation checkpoints and curriculum settings.

**Linux:**
```bash
python scripts/reinforcement_learning/<RL_LIBRARY>/play.py \
    --task=<TASK_NAME>-Play \
    --checkpoint=/absolute/path/to/checkpoint.pth
```

**Windows:**
```cmd
isaaclab.bat -p scripts/reinforcement_learning/<RL_LIBRARY>/play.py ^
    --task=<TASK_NAME>-Play ^
    --checkpoint=C:\absolute\path\to\checkpoint.pth
```

**Note:** On Windows, use forward slashes or escaped backslashes in paths, or use raw strings.

### Demo Scripts

Run example demonstrations and visualizations using trained policies:

**Linux:**
```bash
# Quadruped examples demo
python scripts/demos/quadrupeds.py

# USD policy inference
python scripts/demos/usd_policy_inference.py

# Interactive locomotion demos for specific robots
python scripts/demos/il_anymal_d_usd.py
python scripts/demos/il_go2_rough.py
```

**Windows:**
```cmd
isaaclab.bat -p scripts/demos/quadrupeds.py
isaaclab.bat -p scripts/demos/usd_policy_inference.py
isaaclab.bat -p scripts/demos/il_anymal_d_usd.py
isaaclab.bat -p scripts/demos/il_go2_rough.py
```

**Tip:** Use `isaaclab.sh -p` (Linux) or `isaaclab.bat -p` (Windows) in place of `python` if Isaac Lab is not installed in the active Python environment.

## Multi-Agent Reinforcement Learning

Quadrrl includes support for multi-agent reinforcement learning (MARL) with two distinct task types:

1. **Direct MARL**: Cooperative bar-carrying task with two ANYmal-C robots
2. **Manager-Based MARL**: Velocity tracking task with Spot robot using 4 leg agents

**Note:** MARL tasks are not fully fine-tuned and are still being worked on.

### Setup HARL

HARL is included as a submodule in `scripts/reinforcement_learning/harl/HARL/`. The framework has been customized for Isaac Lab integration and only includes code necessary for Isaac Lab environments.

**HARL Supported Algorithms:**
- `happo` – Hierarchical Actor-Critic PPO (default)
- `hatrpo` – Hierarchical Actor-Critic TRPO
- `haa2c` – Hierarchical Actor-Critic A2C
- `mappo` – Multi-Agent PPO
- `maddpg` – Multi-Agent DDPG
- `matd3` – Multi-Agent TD3
- `hasac` – Hierarchical Actor-Critic SAC
- `hatd3` – Hierarchical Actor-Critic TD3
- `had3qn` – Hierarchical Actor-Critic D3QN
- `haddpg` – Hierarchical Actor-Critic DDPG

### Multi-Agent Task Details

**Template-Quadrrl-MARL-Direct-Anymal-C-v0** (Direct MARL):
- **Agents**: Two ANYmal-C robots
- **Objective**: Cooperatively carry a bar to randomly sampled target locations
- **Observations**: Robot state, joint positions/velocities, target position relative to bar
- **Actions**: Joint position commands (12 DoF per robot)
- **Rewards**: Target distance (primary), target reached bonus, velocity tracking (smoothness)
- **Termination**: Robot falls, bar falls/tilts, or episode timeout
- **Framework**: HARL only

**Template-Quadrrl-Velocity-Flat-Spot-MARL-v0** (Manager-Based MARL):
- **Agents**: Four agents (agentFR, agentFL, agentHR, agentHL) - one per leg
- **Objective**: Velocity tracking on flat terrain
- **Observations**: Base velocity, angular velocity, projected gravity, velocity commands, joint positions/velocities (own leg + other legs), previous actions
- **Actions**: Joint position commands per leg (3 DoF per leg)
- **Rewards**: Velocity tracking, joint regulation, action rate, and other locomotion rewards
- **Termination**: Robot falls or episode timeout
- **Frameworks**: RSL-RL, SKRL, HARL

### Training Multi-Agent Policies

#### ANYmal-C Bar Carrying Task

**Linux:**
```bash
# Train with HAPPO (default)
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=4096 \
    --algorithm=happo \
    --headless

# Train with other algorithms
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=4096 \
    --algorithm=mappo \
    --headless
```

**Windows:**
```cmd
isaaclab.bat -p scripts/reinforcement_learning/harl/train.py ^
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 ^
    --num_envs=4096 ^
    --algorithm=happo ^
    --headless
```

#### Spot Velocity Tracking Task

**Linux:**
```bash
# Using HARL
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 \
    --num_envs=4096 \
    --algorithm=happo \
    --headless

# Using RSL-RL
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 \
    --num_envs=4096

# Using SKRL
python scripts/reinforcement_learning/skrl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 \
    --num_envs=4096
```

**Windows:**
```cmd
REM Using HARL
isaaclab.bat -p scripts/reinforcement_learning/harl/train.py ^
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 ^
    --num_envs=4096 ^
    --algorithm=happo ^
    --headless

REM Using RSL-RL
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py ^
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 ^
    --num_envs=4096

REM Using SKRL
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py ^
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 ^
    --num_envs=4096
```

### Evaluating Multi-Agent Policies

#### ANYmal-C Bar Carrying

**Linux:**
```bash
python scripts/reinforcement_learning/harl/play.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=5 \
    --dir=/path/to/logs/harl/anymal_c_marl/EXPERIMENT_NAME
```

**Windows:**
```cmd
isaaclab.bat -p scripts/reinforcement_learning/harl/play.py ^
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 ^
    --num_envs=5 ^
    --dir=C:\path\to\logs\harl\anymal_c_marl\EXPERIMENT_NAME
```

#### Spot Velocity Tracking

**Linux:**
```bash
# Using HARL
python scripts/reinforcement_learning/harl/play.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 \
    --num_envs=5 \
    --dir=/path/to/logs/harl/spot_marl/EXPERIMENT_NAME

# Using RSL-RL
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 \
    --checkpoint=/path/to/checkpoint.pth

# Using SKRL
python scripts/reinforcement_learning/skrl/play.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 \
    --checkpoint=/path/to/checkpoint.pth
```

**Windows:**
```cmd
REM Using HARL
isaaclab.bat -p scripts/reinforcement_learning/harl/play.py ^
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 ^
    --num_envs=5 ^
    --dir=C:\path\to\logs\harl\spot_marl\EXPERIMENT_NAME

REM Using RSL-RL
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py ^
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 ^
    --checkpoint=C:\path\to\checkpoint.pth

REM Using SKRL
isaaclab.bat -p scripts/reinforcement_learning/skrl/play.py ^
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 ^
    --checkpoint=C:\path\to\checkpoint.pth
```

## Training Tips

1. **Start with fewer environments**: Use `--num_envs=1024` for testing before scaling up
2. **Monitor GPU memory**: Reduce `--num_envs` if you encounter OOM errors
3. **Use headless mode**: Add `--headless` flag for faster training without visualization
4. **Check logs**: Training logs are saved in `logs/<framework>/<task_name>/`
5. **TensorBoard**: Launch TensorBoard to monitor training progress:
   ```bash
   tensorboard --logdir=logs/<framework>/<task_name>/
   ```

## Related Documentation

- [Getting Started Guide](GETTING_STARTED.md) - Basic usage and commands
- [Project Structure](STRUCTURE.md) - Code organization
- [Tasks Documentation](../source/quadrrl/quadrrl/tasks/README.md) - Task implementation details

