# Quadrrl Package

This is the main Quadrrl package, providing robot implementations and task definitions for Isaac Lab.

## Package Structure

```
quadrrl/
├── robots/          # Robot implementations
│   ├── anymal.py   # ANYmal-C and ANYmal-D
│   ├── spot.py     # Spot robot
│   └── unitree.py  # Unitree Go2 and related
└── tasks/          # Task implementations
    ├── direct/     # Direct RL tasks
    └── manager_based/  # Manager-based RL tasks
        └── locomotion/velocity/
            ├── velocity_env_cfg.py      # Quadruped (legged) velocity env
            ├── wheeled_velocity_env_cfg.py  # Wheeled-legged velocity env
            ├── config/
            │   ├── quadrupeds/          # Legged robot configs (Anymal-C/D, Go2, Spot, B2, Lite3, ZSL1)
            │   ├── wheeled/             # Wheeled-legged configs (Go2W, B2W, ZSL1W, M20)
            │   └── spot_marl/           # Spot MARL task config
            └── mdp/                     # MDP components (rewards, terminations, etc.)
```

## Core Components

### Robots

Robot definitions provide:
- Robot asset configurations
- Joint configurations
- Sensor setups
- Initialization parameters

**Available Robots (quadrupeds):**
- **ANYmal-C/D** (`robots/anymal.py`) - ANYbotics quadruped robots
- **Spot** (`robots/spot.py`) - Boston Dynamics Spot robot
- **Unitree Go2, B2, Lite3, ZSL1** (`robots/unitree.py`) - Unitree and related quadrupeds

**Wheeled-legged** velocity configs use the same robot assets with wheeled locomotion (e.g. Unitree Go2W, B2W, Zsibot ZSL1W, DeepRobotics M20).

### Tasks

Tasks define the reinforcement learning environments. See [Tasks Documentation](tasks/README.md) for details.

**Task Types:**
- **Direct Tasks** - Direct RL control without hierarchical structure
- **Manager-Based Tasks** - Hierarchical control with manager and low-level controllers
  - **Quadruped velocity** - Legged locomotion (flat/rough) via `velocity_env_cfg`
  - **Wheeled velocity** - Wheeled-legged locomotion via `wheeled_velocity_env_cfg`

## Extension Registration

Quadrrl is registered as an Isaac Lab extension. The extension configuration is in `config/extension.toml`.

## Usage

Tasks are automatically registered when the package is imported:

```python
import quadrrl.tasks  # Registers all tasks
import gymnasium as gym

env = gym.make("Template-Quadrrl-Velocity-Flat-Anymal-C-v0")  # quadruped
# env = gym.make("Template-Quadrrl-Velocity-Flat-Unitree-Go2W-v0")  # wheeled-legged
```

## Related Documentation

- [Tasks Documentation](tasks/README.md) - Task architecture and implementation
- [Project Structure](../../../docs/STRUCTURE.md) - Overall project organization
- [Training Guide](../../../docs/TRAINING.md) - Training workflows

