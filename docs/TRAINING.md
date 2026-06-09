# Training Guide

## Available Environments

> **Analysis scope:** Performance analysis covers trained **legged** and **wheel-legged** quadruped locomotion policies. Use TensorBoard logs, [`scripts/analysis/analyze_logs.py`](../scripts/analysis/analyze_logs.py), and [`notebooks/rsl_rl_performance.ipynb`](../notebooks/rsl_rl_performance.ipynb) to compare runs and report both policy performance and locomotion task success rate.

### Single-Agent Locomotion Tasks

**Direct Control:**
- `Quadrrl-Velocity-Flat-Anymal-C-Direct-v0`
- `Quadrrl-Velocity-Rough-Anymal-C-Direct-v0`

**Manager-Based Control:**
- ANYmal-C: `Quadrrl-Velocity-Flat-Anymal-C-v0`, `Quadrrl-Velocity-Rough-Anymal-C-v0`
- ANYmal-D: `Quadrrl-Velocity-Flat-Anymal-D-v0`, `Quadrrl-Velocity-Rough-Anymal-D-v0`
- Unitree Go2: `Quadrrl-Velocity-Flat-Unitree-Go2-v0`, `Quadrrl-Velocity-Rough-Unitree-Go2-v0`
- Spot: `Quadrrl-Velocity-Flat-Spot-v0`, `Quadrrl-Velocity-Rough-Spot-v0`

**Note:** Spot uses gait- and contact-focused rewards (gait phase shaping, foot-clearance, air-time balance) that differ from the generic locomotion reward set used by ANYmal/Go2.

**Legged (velocity):**  
Environments use the Isaac Lab-style `locomotion/legged/velocity_env_cfg` base and are registered under `locomotion/legged/config/`.

**Wheeled-legged (velocity only):**  
Environments use the `locomotion/wheeled/velocity_env_cfg` base and are registered under `locomotion/wheeled/config/`. Examples: `Quadrrl-Velocity-Flat-Unitree-Go2W-v0`, `Quadrrl-Velocity-Rough-Unitree-Go2W-v0`, and similarly for Unitree B2W, Zsibot ZSL1W, and DeepRobotics M20. Use the same training/eval commands with the corresponding task name.

### Single-Agent Navigation Tasks

- `Quadrrl-Navigation-Flat-Anymal-C-v0`
- `Quadrrl-Navigation-Rough-Anymal-C-v0`

### Multi-Agent Tasks

- `Quadrrl-MARL-Direct-Anymal-C-v0` - Cooperative bar-carrying (HARL)

> **Note:** HARL remains supported for multi-agent RL algorithms. The previous Spot-MARL integration has been removed. MARL tasks are still being tuned. Use `scripts/list_envs.py` to see all available environments.

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
    --task=Quadrrl-Velocity-Flat-Anymal-C-v0 \
    --num_envs=4096 --seed=42
```

### Evaluation

```bash
python scripts/reinforcement_learning/<RL_LIBRARY>/play.py \
    --task=<TASK_NAME> \
    --checkpoint=/absolute/path/to/checkpoint.pth
```

When reporting results, include both policy performance and locomotion task success rate for trained legged and wheel-legged quadruped robots.

### Demo Scripts

```bash
python scripts/demos/quadrupeds.py
python scripts/demos/usd_policy_inference.py
python scripts/demos/il_anymal_d_usd.py
python scripts/demos/il_go2_rough.py
```

**Tip:** Use `isaaclab.sh -p` (Linux) or `isaaclab.bat -p` (Windows) if Isaac Lab is not in your Python PATH.

## Multi-Agent Reinforcement Learning

Quadrrl currently includes one MARL task type:
1. **Direct MARL**: Cooperative bar-carrying task with two ANYmal-C robots

**Note:** MARL tasks are not fully fine-tuned and are still being worked on.

### Setup HARL

HARL is included as a submodule in `scripts/reinforcement_learning/harl/HARL/` and has been customized for Isaac Lab integration.

**HARL Supported Algorithms:** `happo` (default), `hatrpo`, `haa2c`, `mappo`, `maddpg`, `matd3`, `hasac`, `hatd3`, `had3qn`, `haddpg`

### Multi-Agent Task Details

**Quadrrl-MARL-Direct-Anymal-C-v0** (Direct MARL):
- **Agents**: Two ANYmal-C robots
- **Objective**: Cooperatively carry a bar to randomly sampled target locations
- **Framework**: HARL only

### Training Multi-Agent Policies

**ANYmal-C Bar Carrying Task:**
```bash
python scripts/reinforcement_learning/harl/train.py \
    --task=Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=4096 --algorithm=happo --headless
```

### Evaluating Multi-Agent Policies

**ANYmal-C Bar Carrying:**
```bash
python scripts/reinforcement_learning/harl/play.py \
    --task=Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=5 --dir=/path/to/logs/harl/anymal_c_marl/EXPERIMENT_NAME
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
- [Notebooks](../notebooks/README.md) - Performance analysis with `rsl_rl_performance.ipynb`
