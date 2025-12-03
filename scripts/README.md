# Scripts Documentation

This directory contains utility scripts, training scripts, and demonstration scripts for Quadrrl.

## Directory Structure

```
scripts/
├── analysis/                    # Log analysis utilities
│   ├── analyze_logs.py         # Main analysis script
│   └── rsl_rl_analysis_utils.py # RSL-RL specific utilities
├── demos/                      # Demonstration scripts
│   ├── quadrupeds.py           # General quadruped demos
│   ├── usd_policy_inference.py # USD-based policy inference
│   ├── il_anymal_d_usd.py      # ANYmal-D interactive locomotion
│   └── il_go2_rough.py         # Go2 rough terrain demo
├── reinforcement_learning/     # RL framework training scripts
│   ├── rl_games/              # RL Games framework
│   ├── rsl_rl/                # RSL-RL framework
│   ├── skrl/                  # SKRL framework
│   └── harl/                  # HARL multi-agent framework
├── list_envs.py               # List all registered environments
├── check_assets.py            # Verify asset files
├── random_agent.py            # Random agent baseline
└── zero_agent.py              # Zero action baseline
```

## Utility Scripts

### list_envs.py

List all available Quadrrl environments.

**Usage:**
```bash
# Linux
python scripts/list_envs.py

# Windows
isaaclab.bat -p scripts/list_envs.py
```

**Output:** Table showing task names, entry points, and configuration files.

### check_assets.py

Verify that required asset files are present.

**Usage:**
```bash
# Linux
python scripts/check_assets.py

# Windows
isaaclab.bat -p scripts/check_assets.py
```

### random_agent.py

Run a random agent baseline for testing environments.

**Usage:**
```bash
# Linux
python scripts/random_agent.py --task=<TASK_NAME>

# Windows
isaaclab.bat -p scripts/random_agent.py --task=<TASK_NAME>
```

### zero_agent.py

Run a zero-action baseline for testing environments.

**Usage:**
```bash
# Linux
python scripts/zero_agent.py --task=<TASK_NAME>

# Windows
isaaclab.bat -p scripts/zero_agent.py --task=<TASK_NAME>
```

## Analysis Scripts

### analyze_logs.py

Analyze training logs and generate performance reports.

**Usage:**
```bash
# Linux
python scripts/analysis/analyze_logs.py

# Windows
isaaclab.bat -p scripts/analysis/analyze_logs.py
```

**Features:**
- Extract training metrics
- Compare different experiments
- Generate performance plots
- Export analysis reports

### rsl_rl_analysis_utils.py

Utilities for RSL-RL log analysis. Used by analysis notebooks and scripts.

**Key Functions:**
- `load_all_metrics()` - Load metrics from TensorBoard logs
- `extract_key_metrics()` - Extract specific metrics
- `plot_comparison_bar()` - Generate comparison plots
- `plot_training_curves()` - Plot training progress

## Demo Scripts

### quadrupeds.py

General quadruped demonstration script showcasing various robots and tasks.

**Usage:**
```bash
# Linux
python scripts/demos/quadrupeds.py

# Windows
isaaclab.bat -p scripts/demos/quadrupeds.py
```

### usd_policy_inference.py

Run policy inference using USD scene files.

**Usage:**
```bash
# Linux
python scripts/demos/usd_policy_inference.py \
    --checkpoint logs/rsl_rl/anymal_c_flat/EXPERIMENT_NAME/exported/policy.pt

# Windows
isaaclab.bat -p scripts/demos/usd_policy_inference.py ^
    --checkpoint logs\rsl_rl\anymal_c_flat\EXPERIMENT_NAME\exported\policy.pt
```

### il_anymal_d_usd.py

Interactive locomotion demo for ANYmal-D robot.

**Usage:**
```bash
# Linux
python scripts/demos/il_anymal_d_usd.py

# Windows
isaaclab.bat -p scripts/demos/il_anymal_d_usd.py
```

### il_go2_rough.py

Interactive locomotion demo for Unitree Go2 on rough terrain.

**Usage:**
```bash
# Linux
python scripts/demos/il_go2_rough.py

# Windows
isaaclab.bat -p scripts/demos/il_go2_rough.py
```

## Reinforcement Learning Scripts

### Training Scripts

Each RL framework has its own `train.py` script:

- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rl_games/train.py`
- `scripts/reinforcement_learning/skrl/train.py`
- `scripts/reinforcement_learning/harl/train.py`

**Common Usage:**
```bash
# Linux
python scripts/reinforcement_learning/<FRAMEWORK>/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096

# Windows
isaaclab.bat -p scripts/reinforcement_learning/<FRAMEWORK>/train.py ^
    --task=<TASK_NAME> ^
    --num_envs=4096
```

See [Training Guide](../docs/TRAINING.md) for detailed usage.

### Evaluation Scripts

Each RL framework has its own `play.py` script for evaluation:

- `scripts/reinforcement_learning/rsl_rl/play.py`
- `scripts/reinforcement_learning/rl_games/play.py`
- `scripts/reinforcement_learning/skrl/play.py`
- `scripts/reinforcement_learning/harl/play.py`

**Common Usage:**
```bash
# Linux
python scripts/reinforcement_learning/<FRAMEWORK>/play.py \
    --task=<TASK_NAME>-Play \
    --checkpoint=/path/to/checkpoint.pth

# Windows
isaaclab.bat -p scripts/reinforcement_learning/<FRAMEWORK>/play.py ^
    --task=<TASK_NAME>-Play ^
    --checkpoint=C:\path\to\checkpoint.pth
```

## Related Documentation

- [Getting Started Guide](../docs/GETTING_STARTED.md) - Basic usage
- [Training Guide](../docs/TRAINING.md) - Detailed training instructions
- [Project Structure](../docs/STRUCTURE.md) - Code organization

