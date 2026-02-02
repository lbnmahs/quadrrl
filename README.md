# QUADRRL

![Quadrrl Header](assets/QUADRRL.png)

## Overview

Quadrrl builds on NVIDIA Isaac Lab to research and prototype deep reinforcement learning for quadruped robots. It includes locomotion and navigation tasks across flat and rough terrains, covering velocity tracking, goal-directed navigation, and multi-agent coordination.

**Highlights**
- Unified training suite for ANYmal-C, ANYmal-D, Spot, and Unitree Go2 robots
- Direct and manager-based task variants for locomotion and navigation
- Ready-to-run configs for `rl_games`, `rsl_rl`, `skrl`, and `harl` RL frameworks
- Multi-agent reinforcement learning (MARL) support for cooperative tasks
- Spot uses a tailored reward structure (gait shaping, foot clearance, air-time balancing) that differs from generic locomotion rewards, enabling richer gait coordination experiments

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Setup instructions
- **[Getting Started](docs/GETTING_STARTED.md)** - Quick start guide
- **[Simulation Videos](docs/DEMOS.md)** - Video demonstrations
- **[Project Structure](docs/STRUCTURE.md)** - Code organization
- **[Training Guide](docs/TRAINING.md)** - Single-agent and multi-agent RL training
- **[Scripts Documentation](scripts/README.md)** - Available scripts and utilities
- **[Tasks Documentation](source/quadrrl/quadrrl/tasks/README.md)** - Task architecture

## 🚀 Quick Start

```bash
git clone https://github.com/lbnmahs/quadrrl.git
cd quadrrl
conda activate isaaclab
python -m pip install -e source/quadrrl
python scripts/list_envs.py
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-v0 \
    --num_envs=4096
```

See [Installation Guide](docs/INSTALLATION.md) and [Getting Started Guide](docs/GETTING_STARTED.md) for details.

## 🤖 Supported Robots

- **ANYmal-C** - Direct and manager-based locomotion, navigation, MARL
- **ANYmal-D** - Manager-based locomotion on flat and rough terrain
- **Spot** - Manager-based locomotion and MARL velocity tracking (distinct gait-focused reward structure)
- **Unitree Go2** - Manager-based locomotion on flat and rough terrain

## 🎯 Available Tasks

### Single-Agent RL
**RSL-RL analyses focus on:** Unitree Go2, ANYmal-C, ANYmal-D
- Velocity tracking (flat and rough terrain)
- Goal-directed navigation
- Direct and manager-based control

### Multi-Agent RL
- Cooperative bar-carrying (ANYmal-C)
- Multi-agent velocity tracking (Spot with 4 leg agents)
  - Spot is used primarily to compare single-agent vs multi-agent RL with HARL

**Note:** MARL tasks are not fully fine-tuned and are still being worked on.

See [Training Guide](docs/TRAINING.md) for complete task listings.

## 🔧 RL Frameworks

- **RSL-RL** - Default framework with PPO support
- **RL Games** - NVIDIA's RL framework
- **SKRL** - Scikit-learn compatible RL library
- **HARL** - Multi-agent RL framework (customized for Isaac Lab)

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16 GB minimum, 32 GB recommended
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11 (64-bit)
- **Isaac Lab**: Installed per [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python**: 3.10 or newer

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to Quadrrl.

- Fork the repository, create feature branches, and open pull requests with clear descriptions.
- Run `pre-commit run --all-files` before submitting changes.
- Add tests or evaluation scripts when introducing new environments or reward structures.
- Update documentation when you add new tasks or major capabilities.

## 📖 Citation

If you use Quadrrl in your research, please cite:

```bibtex
@software{quadrrl2026,
  title={Quadrrl: Isaac Lab-Based Multi-Quadruped Locomotion Training and Performance Evaluation Suite},
  author={Mahihu, Laban Njoroge},
  year={2026},
  url={https://github.com/lbnmahs/quadrrl}
}
```

## 📄 License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to Dr. Manal Helal for her guidance and advice. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for a complete list of acknowledgments.

## 📚 Resources & Inspiration

- [NVIDIA Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [HARL Framework](https://github.com/PKU-MARL/HARL)