# Quadrrl Package

This is the main Quadrrl package, providing robot implementations and task definitions for Isaac Lab.

## Package Structure

```
quadrrl/
├── robots/          # Robot implementations
│   ├── anymal.py   # ANYmal-C and ANYmal-D
│   ├── spot.py     # Spot robot
│   └── unitree.py  # Unitree Go2
└── tasks/          # Task implementations
    ├── direct/     # Direct RL tasks
    └── manager_based/  # Manager-based RL tasks
```

## Core Components

### Robots

Robot definitions provide:
- Robot asset configurations
- Joint configurations
- Sensor setups
- Initialization parameters

**Available Robots:**
- **ANYmal-C/D** (`robots/anymal.py`) - ANYbotics quadruped robots
- **Spot** (`robots/spot.py`) - Boston Dynamics Spot robot
- **Unitree Go2** (`robots/unitree.py`) - Unitree Go2 quadruped

### Tasks

Tasks define the reinforcement learning environments. See [Tasks Documentation](tasks/README.md) for details.

**Task Types:**
- **Direct Tasks** - Direct RL control without hierarchical structure
- **Manager-Based Tasks** - Hierarchical control with manager and low-level controllers

## Extension Registration

Quadrrl is registered as an Isaac Lab extension. The extension configuration is in `config/extension.toml`.

## Usage

Tasks are automatically registered when the package is imported:

```python
import quadrrl.tasks  # Registers all tasks
import gymnasium as gym

env = gym.make("Template-Quadrrl-Velocity-Flat-Anymal-C-v0")
```

## Related Documentation

- [Tasks Documentation](tasks/README.md) - Task architecture and implementation
- [Project Structure](../../../docs/STRUCTURE.md) - Overall project organization
- [Training Guide](../../../docs/TRAINING.md) - Training workflows

