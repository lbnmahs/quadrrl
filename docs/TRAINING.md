# Training Guide

## Available Environments

> **Analysis scope:** RSL-RL benchmark analysis focuses on **Unitree Go2**, **ANYmal-C**, and **ANYmal-D** for single-agent velocity-tracking. Spot single-agent runs remain available but are not part of the core RSL-RL comparison; Spot is primarily used for single-vs-multi-agent comparisons with HARL.

### Single-Agent Locomotion Tasks

**Direct Control:**
- `Template-Quadrrl-Velocity-Flat-Anymal-C-Direct-v0`
- `Template-Quadrrl-Velocity-Rough-Anymal-C-Direct-v0`

**Manager-Based Control:**
- ANYmal-C: `Template-Quadrrl-Velocity-Flat-Anymal-C-v0`, `Template-Quadrrl-Velocity-Rough-Anymal-C-v0` (+ `-Play` variants)
- ANYmal-D: `Template-Quadrrl-Velocity-Flat-Anymal-D-v0`, `Template-Quadrrl-Velocity-Rough-Anymal-D-v0` (+ `-Play` variants)
- Unitree Go2: `Template-Quadrrl-Velocity-Flat-Unitree-Go2-v0`, `Template-Quadrrl-Velocity-Rough-Unitree-Go2-v0` (+ `-Play` variants)
- Spot: `Template-Quadrrl-Velocity-Flat-Spot-v0`, `Template-Quadrrl-Velocity-Rough-Spot-v0` (+ `-Play` variants)

**Note:** Spot uses gait- and contact-focused rewards (gait phase shaping, foot-clearance, air-time balance) that differ from the generic locomotion reward set used by ANYmal/Go2.

### Single-Agent Navigation Tasks

- `Template-Quadrrl-Navigation-Flat-Anymal-C-v0`
- `Template-Quadrrl-Navigation-Rough-Anymal-C-v0` (+ `-Play` variants)

### Multi-Agent Tasks

- `Template-Quadrrl-MARL-Direct-Anymal-C-v0` - Cooperative bar-carrying (HARL)
- `Template-Quadrrl-Velocity-Flat-Spot-MARL-v0` - Velocity tracking with 4 leg agents (HARL primary, RSL-RL optional) (+ `-Play` variant)

> **Comparison plan:** Spot is the primary benchmark for single-agent vs multi-agent RL using HARL (leg-level agents). RSL-RL analysis remains focused on Unitree Go2, ANYmal-C, and ANYmal-D for single-agent baselines.

**Note:** MARL tasks are not fully fine-tuned and are still being worked on. Use `scripts/list_envs.py` to see all available environments.

## Single-Agent Reinforcement Learning

### Training

Replace `<RL_LIBRARY>` with `rl_games`, `rsl_rl`, `skrl`, or `harl`.

```bash
# Linux
python scripts/reinforcement_learning/<RL_LIBRARY>/train.py \
    --task=<TASK_NAME> --num_envs=4096 --seed=42

# Windows
isaaclab.bat -p scripts/reinforcement_learning/<RL_LIBRARY>/train.py ^
    --task=<TASK_NAME> --num_envs=4096 --seed=42
```

**Example:**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-v0 \
    --num_envs=4096 --seed=42
```

### Evaluation

Add the suffix `-Play` to load evaluation checkpoints and curriculum settings.

```bash
python scripts/reinforcement_learning/<RL_LIBRARY>/play.py \
    --task=<TASK_NAME>-Play \
    --checkpoint=/absolute/path/to/checkpoint.pth
```

### Demo Scripts

```bash
python scripts/demos/quadrupeds.py
python scripts/demos/usd_policy_inference.py
python scripts/demos/il_anymal_d_usd.py
python scripts/demos/il_go2_rough.py
```

**Tip:** Use `isaaclab.sh -p` (Linux) or `isaaclab.bat -p` (Windows) if Isaac Lab is not in your Python PATH.

## Multi-Agent Reinforcement Learning

Quadrrl includes two MARL task types:
1. **Direct MARL**: Cooperative bar-carrying task with two ANYmal-C robots
2. **Manager-Based MARL**: Velocity tracking task with Spot robot using 4 leg agents

**Note:** MARL tasks are not fully fine-tuned and are still being worked on.

### Setup HARL

HARL is included as a submodule in `scripts/reinforcement_learning/harl/HARL/` and has been customized for Isaac Lab integration.

**HARL Supported Algorithms:** `happo` (default), `hatrpo`, `haa2c`, `mappo`, `maddpg`, `matd3`, `hasac`, `hatd3`, `had3qn`, `haddpg`

### Multi-Agent Task Details

**Template-Quadrrl-MARL-Direct-Anymal-C-v0** (Direct MARL):
- **Agents**: Two ANYmal-C robots
- **Objective**: Cooperatively carry a bar to randomly sampled target locations
- **Framework**: HARL only

**Template-Quadrrl-Velocity-Flat-Spot-MARL-v0** (Manager-Based MARL):
- **Agents**: Four agents (agentFR, agentFL, agentHR, agentHL) - one per leg
- **Objective**: Velocity tracking on flat terrain
- **Frameworks**: HARL (primary), RSL-RL, SKRL

### Training Multi-Agent Policies

**ANYmal-C Bar Carrying Task:**
```bash
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=4096 --algorithm=happo --headless
```

**Spot Velocity Tracking Task:**
```bash
# Using HARL
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 \
    --num_envs=4096 --algorithm=happo --headless

# Using RSL-RL
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 \
    --num_envs=4096

# Using SKRL
python scripts/reinforcement_learning/skrl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-v0 \
    --num_envs=4096
```

### Evaluating Multi-Agent Policies

**ANYmal-C Bar Carrying:**
```bash
python scripts/reinforcement_learning/harl/play.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=5 --dir=/path/to/logs/harl/anymal_c_marl/EXPERIMENT_NAME
```

**Spot Velocity Tracking:**
```bash
# Using HARL
python scripts/reinforcement_learning/harl/play.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 \
    --num_envs=5 --dir=/path/to/logs/harl/spot_marl/EXPERIMENT_NAME

# Using RSL-RL or SKRL
python scripts/reinforcement_learning/<RL_LIBRARY>/play.py \
    --task=Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0 \
    --checkpoint=/path/to/checkpoint.pth
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
