# Project Structure

This document describes the organization of the Quadrrl codebase.

## Directory Layout

```
quadrrl/
├── assets/                 # Project assets (images, etc.)
├── docs/                   # Documentation files
│   ├── INSTALLATION.md
│   ├── GETTING_STARTED.md
│   ├── STRUCTURE.md
│   └── TRAINING.md
├── logs/                   # Training logs and checkpoints
│   ├── rsl_rl/            # RSL-RL training logs
│   ├── skrl/              # SKRL training logs
│   ├── harl/              # HARL training logs
│   └── spot_marl/         # Multi-agent logs
├── notebooks/             # Analysis notebooks
│   ├── README.md
│   └── rsl_rl_performance.ipynb
├── outputs/               # Hydra output logs
├── scripts/               # Utility and training scripts
│   ├── analysis/         # Log analysis utilities
│   ├── demos/            # Demonstration scripts
│   └── reinforcement_learning/  # RL framework scripts
│       ├── rl_games/
│       ├── rsl_rl/
│       ├── skrl/
│       └── harl/
├── source/                # Source code
│   └── quadrrl/
│       ├── config/        # Extension configuration
│       ├── docs/          # Package documentation
│       └── quadrrl/       # Main package
│           ├── robots/    # Robot implementations
│           └── tasks/     # Task implementations
└── README.md              # Main project README
```

## Core Components

### Source Code (`source/quadrrl/quadrrl/`)

#### Robots (`robots/`)
Robot-specific implementations:
- `anymal.py` - ANYmal-C and ANYmal-D robot definitions
- `spot.py` - Spot robot definition
- `unitree.py` - Unitree Go2 robot definition

#### Tasks (`tasks/`)
Task implementations organized by architecture:

**Direct Tasks** (`tasks/direct/`)
- `anymal_c/` - Direct ANYmal-C locomotion
- `anymal_c_marl/` - Direct multi-agent ANYmal-C bar-carrying

**Manager-Based Tasks** (`tasks/manager_based/`)
- `locomotion/velocity/` - Velocity tracking tasks
  - `config/` - Robot-specific configurations
    - `anymal_c/`, `anymal_d/`, `go2/`, `spot/`, `spot_marl/`
  - `mdp/` - MDP components (rewards, terminations, curriculums)
- `navigation/` - Navigation tasks
  - `config/anymal_c/` - ANYmal-C navigation configs
  - `mdp/` - Navigation-specific MDP components

Each task directory contains:
- Environment implementation (`*_env.py`)
- Configuration (`*_env_cfg.py`)
- Agent configs (`agents/`) - RL framework-specific configurations

### Scripts (`scripts/`)

#### Training Scripts (`reinforcement_learning/`)
Framework-specific training and evaluation:
- `rl_games/` - RL Games framework
- `rsl_rl/` - RSL-RL framework
- `skrl/` - SKRL framework
- `harl/` - HARL multi-agent framework

#### Analysis Scripts (`analysis/`)
- `analyze_logs.py` - Log analysis utilities
- `rsl_rl_analysis_utils.py` - RSL-RL specific analysis

#### Demo Scripts (`demos/`)
- `quadrupeds.py` - General quadruped demos
- `usd_policy_inference.py` - USD-based policy inference
- `il_anymal_d_usd.py` - ANYmal-D interactive locomotion
- `il_go2_rough.py` - Go2 rough terrain demo

#### Utility Scripts
- `list_envs.py` - List all registered environments
- `check_assets.py` - Verify asset files
- `random_agent.py` - Random agent baseline
- `zero_agent.py` - Zero action baseline

### Logs (`logs/`)

Training logs organized by framework and task:
- `rsl_rl/<task_name>/<experiment_date>/` - RSL-RL experiments
- `skrl/<task_name>/<experiment_date>/` - SKRL experiments
- `harl/<task_name>/<experiment_date>/` - HARL experiments

Each experiment directory contains:
- Checkpoints (`model_*.pt`)
- TensorBoard logs
- Configuration files
- Training metrics

## Code Organization Principles

1. **Modularity**: Tasks are self-contained with their own configs
2. **Extensibility**: Easy to add new robots or tasks
3. **Framework Agnostic**: Task implementations work with multiple RL frameworks
4. **Separation of Concerns**: Robots, tasks, and RL frameworks are decoupled

## File Naming Conventions

- **Environments**: `*_env.py` (e.g., `anymal_c_env.py`)
- **Configurations**: `*_env_cfg.py` (e.g., `flat_env_cfg.py`)
- **Agent Configs**: `<framework>_<algorithm>_cfg.yaml` or `.py` (e.g., `rsl_rl_ppo_cfg.py`)
- **MDP Components**: Descriptive names (e.g., `rewards.py`, `terminations.py`)

## Adding New Components

### Adding a New Robot
1. Create robot file in `source/quadrrl/quadrrl/robots/`
2. Define robot configuration class
3. Register in appropriate task configs

### Adding a New Task
1. Create task directory in `tasks/direct/` or `tasks/manager_based/`
2. Implement environment class (`*_env.py`)
3. Create configuration (`*_env_cfg.py`)
4. Add agent configs for desired RL frameworks
5. Register environment in `tasks/__init__.py`

### Adding a New RL Framework
1. Create framework directory in `scripts/reinforcement_learning/`
2. Implement `train.py` and `play.py`
3. Create agent config templates
4. Update documentation

## Related Documentation

- [Tasks Documentation](../source/quadrrl/quadrrl/tasks/README.md) - Detailed task architecture
- [Scripts Documentation](../scripts/README.md) - Script usage and examples
- [Training Guide](TRAINING.md) - Training workflows

