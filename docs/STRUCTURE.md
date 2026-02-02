# Project Structure

## Directory Layout

```
quadrrl/
├── assets/                 # Project assets
├── docs/                   # Documentation
├── logs/                   # Training logs and checkpoints
│   ├── rsl_rl/, skrl/, harl/
├── notebooks/             # Analysis notebooks
├── outputs/               # Hydra output logs
├── scripts/               # Utility and training scripts
│   ├── analysis/         # Log analysis utilities
│   ├── demos/            # Demonstration scripts
│   └── reinforcement_learning/  # RL framework scripts
│       ├── rl_games/, rsl_rl/, skrl/, harl/
├── source/quadrrl/quadrrl/  # Main package
│   ├── robots/           # Robot implementations
│   └── tasks/            # Task implementations
│       ├── direct/       # Direct control tasks
│       └── manager_based/  # Manager-based tasks
└── README.md
```

## Core Components

### Source Code (`source/quadrrl/quadrrl/`)

**Robots** (`robots/`): `anymal.py`, `spot.py`, `unitree.py`

**Tasks** (`tasks/`):
- **Direct** (`tasks/direct/`): `anymal_c/`, `anymal_c_marl/`
- **Manager-Based** (`tasks/manager_based/`):
  - `locomotion/velocity/` - Velocity tracking with robot-specific configs
  - `navigation/` - Navigation tasks

Each task contains: environment (`*_env.py`), config (`*_env_cfg.py`), and agent configs (`agents/`)

### Scripts (`scripts/`)

**Training** (`reinforcement_learning/`): `rl_games/`, `rsl_rl/`, `skrl/`, `harl/`  
**Analysis** (`analysis/`): Log analysis utilities  
**Demos** (`demos/`): Demonstration scripts  
**Utilities**: `list_envs.py`, `check_assets.py`, `random_agent.py`, `zero_agent.py`

### Logs (`logs/`)

Organized by framework and task: `logs/<framework>/<task_name>/<experiment_date>/`  
Contains: checkpoints, TensorBoard logs, configs, metrics

## Code Organization Principles

1. **Modularity**: Tasks are self-contained with their own configs
2. **Extensibility**: Easy to add new robots or tasks
3. **Framework Agnostic**: Task implementations work with multiple RL frameworks
4. **Separation of Concerns**: Robots, tasks, and RL frameworks are decoupled

## File Naming Conventions

- **Environments**: `*_env.py`
- **Configurations**: `*_env_cfg.py`
- **Agent Configs**: `<framework>_<algorithm>_cfg.yaml` or `.py`
- **MDP Components**: Descriptive names (e.g., `rewards.py`, `terminations.py`)

## Adding New Components

**New Robot**: Create file in `robots/`, define config class, register in task configs  
**New Task**: Create directory in `tasks/direct/` or `tasks/manager_based/`, implement environment and config, add agent configs, register in `tasks/__init__.py`  
**New RL Framework**: Create directory in `scripts/reinforcement_learning/`, implement `train.py` and `play.py`, create agent config templates

## Related Documentation

- [Tasks Documentation](../source/quadrrl/quadrrl/tasks/README.md) - Task architecture
- [Scripts Documentation](../scripts/README.md) - Script usage
- [Training Guide](TRAINING.md) - Training workflows

