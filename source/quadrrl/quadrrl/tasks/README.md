# Tasks Documentation

This directory contains all task implementations for Quadrrl, organized by architecture type.

## Task Architecture

### Direct Tasks (`direct/`)

Direct RL tasks where the policy directly controls robot joints without hierarchical structure.

**Available Direct Tasks:**
- `anymal_c/` - Direct ANYmal-C locomotion (flat and rough terrain)
- `anymal_c_marl/` - Direct multi-agent ANYmal-C bar-carrying

**Structure:**
```
direct/
└── <task_name>/
    ├── <task_name>_env.py        # Environment implementation
    ├── <task_name>_env_cfg.py     # Environment configuration
    └── agents/                    # RL framework configs
        ├── rsl_rl_ppo_cfg.py
        ├── skrl_flat_ppo_cfg.yaml
        └── ...
```

### Manager-Based Tasks (`manager_based/`)

Manager-based tasks use hierarchical control with a manager (high-level) and low-level controllers.

**Available Manager-Based Tasks:**
- `locomotion/velocity/` - Velocity tracking tasks
- `navigation/` - Goal-directed navigation tasks

**Structure:**
```
manager_based/
├── locomotion/
│   └── velocity/
│       ├── velocity_env_cfg.py   # Base velocity task config
│       ├── config/                # Robot-specific configs
│       │   ├── anymal_c/
│       │   ├── anymal_d/
│       │   ├── go2/
│       │   ├── spot/
│       │   └── spot_marl/
│       └── mdp/                   # MDP components
│           ├── rewards.py
│           ├── terminations.py
│           ├── curriculums.py
│           └── symmetry/
└── navigation/
    └── config/
        └── anymal_c/
```

## Task Components

### Environment Implementation (`*_env.py`)

Defines the Gymnasium environment class:
- Observation space
- Action space
- Reset logic
- Step function
- Reward computation
- Termination conditions

### Environment Configuration (`*_env_cfg.py`)

Configuration class specifying:
- Robot configuration
- Scene settings
- Observation/action spaces
- Reward weights
- Curriculum parameters
- Termination conditions

### Agent Configurations (`agents/`)

RL framework-specific configuration files:
- `rsl_rl_ppo_cfg.py` - RSL-RL PPO configuration
- `skrl_*_ppo_cfg.yaml` - SKRL configurations
- `rl_games_*_ppo_cfg.yaml` - RL Games configurations
- `harl_*_cfg.yaml` - HARL algorithm configurations

### MDP Components (`mdp/`)

Markov Decision Process components:
- **Rewards** (`rewards.py`) - Reward function implementations
- **Terminations** (`terminations.py`) - Episode termination conditions
- **Curriculums** (`curriculums.py`) - Curriculum learning schedules
- **Symmetry** (`symmetry/`) - Symmetry-based data augmentation

## Task Categories

### Locomotion Tasks

Locomotion tasks focus on velocity tracking and basic movement:
- **Direct Locomotion**: Direct control of joint positions
- **Manager-Based Locomotion**: Hierarchical control with velocity commands

**Available Robots:**
- ANYmal-C (direct and manager-based)
- ANYmal-D (manager-based)
- Unitree Go2 (manager-based)
- Spot (manager-based, single-agent and multi-agent)

### Navigation Tasks

Navigation tasks involve goal-directed movement:
- **Manager-Based Navigation**: Hierarchical control with goal positions

**Available Robots:**
- ANYmal-C (manager-based, flat and rough terrain)

### Multi-Agent Tasks

Multi-agent tasks involve coordination between multiple agents:
- **Direct MARL**: Two ANYmal-C robots cooperatively carrying a bar
- **Manager-Based MARL**: Spot robot with 4 leg agents for velocity tracking

**Note:** MARL tasks are not fully fine-tuned and are still being worked on.

## Adding a New Task

### Step 1: Choose Task Type

Decide between:
- **Direct Task**: Simpler, direct control
- **Manager-Based Task**: Hierarchical control structure

### Step 2: Create Task Directory

```bash
# For direct task
mkdir -p source/quadrrl/quadrrl/tasks/direct/<task_name>

# For manager-based task
mkdir -p source/quadrrl/quadrrl/tasks/manager_based/<category>/<task_name>
```

### Step 3: Implement Environment

Create `*_env.py` with:
- Environment class inheriting from `DirectRLEnv` or `ManagerBasedRLEnv`
- Required methods: `_design_scene()`, `_design_actions()`, `_design_observations()`, etc.

### Step 4: Create Configuration

Create `*_env_cfg.py` with:
- Configuration class inheriting from `DirectRLEnvCfg` or `ManagerBasedRLEnvCfg`
- Robot configuration
- Task-specific parameters

### Step 5: Add Agent Configs

Create framework-specific agent configurations in `agents/` directory.

### Step 6: Register Environment

The environment is automatically registered when the package is imported if:
- The config class is properly named
- The `env_cfg_entry_point` is correctly specified
- The package is imported via `import quadrrl.tasks`

### Step 7: Update Documentation

- Add task to environment listing in `scripts/list_envs.py`
- Update [Training Guide](../../../../docs/TRAINING.md)
- Add task description to this README

## Task Naming Convention

Tasks follow the pattern:
```
Template-Quadrrl-<TaskType>-<Terrain>-<Robot>-<Variant>-v0
```

Examples:
- `Template-Quadrrl-Velocity-Flat-Anymal-C-v0`
- `Template-Quadrrl-Velocity-Rough-Unitree-Go2-v0`
- `Template-Quadrrl-MARL-Direct-Anymal-C-v0`
- `Template-Quadrrl-Navigation-Flat-Anymal-C-v0`

## Related Documentation

- [Project Structure](../../../../docs/STRUCTURE.md) - Overall project organization
- [Training Guide](../../../../docs/TRAINING.md) - Training workflows
- [Isaac Lab Tasks Documentation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/task_creation/index.html)

