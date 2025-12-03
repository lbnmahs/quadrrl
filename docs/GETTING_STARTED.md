# Getting Started

This guide will help you get started with Quadrrl, from listing environments to running your first training session.

## Prerequisites

- Quadrrl installed (see [Installation Guide](INSTALLATION.md))
- Isaac Lab environment activated
- GPU with CUDA support

## Running Scripts

### Linux

Most scripts can be run directly with Python:

```bash
python scripts/list_envs.py
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME>
```

If Isaac Lab is not in your Python PATH, use the Isaac Lab launcher:

```bash
isaaclab.sh -p scripts/list_envs.py
```

### Windows

On Windows, use the Isaac Lab batch file:

```cmd
isaaclab.bat -p scripts/list_envs.py
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME>
```

**Note**: Replace `python` with `isaaclab.bat -p` for all commands on Windows if Isaac Lab is not in your PATH.

## Listing Available Environments

View all available Quadrrl environments:

```bash
# Linux
python scripts/list_envs.py

# Windows
isaaclab.bat -p scripts/list_envs.py
```

This displays a table with:
- Task names
- Environment entry points
- Configuration files

See [Training Guide](TRAINING.md) for detailed task descriptions.

## Quick Training Example

Train a single-agent velocity tracking policy:

```bash
# Linux
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-v0 \
    --num_envs=4096 \
    --seed=42

# Windows
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py ^
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-v0 ^
    --num_envs=4096 ^
    --seed=42
```

**Note**: On Windows, use `^` for line continuation in Command Prompt, or use `\` in PowerShell.

## Running Demo Scripts

Explore pre-built demonstrations:

```bash
# Linux
python scripts/demos/quadrupeds.py
python scripts/demos/usd_policy_inference.py
python scripts/demos/il_anymal_d_usd.py
python scripts/demos/il_go2_rough.py

# Windows
isaaclab.bat -p scripts/demos/quadrupeds.py
isaaclab.bat -p scripts/demos/usd_policy_inference.py
isaaclab.bat -p scripts/demos/il_anymal_d_usd.py
isaaclab.bat -p scripts/demos/il_go2_rough.py
```

## Common Commands

### Training
```bash
# Single-agent training
python scripts/reinforcement_learning/<RL_LIBRARY>/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --seed=42

# Multi-agent training (HARL)
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=4096 \
    --algorithm=happo \
    --headless
```

### Evaluation
```bash
# Play with saved checkpoint
python scripts/reinforcement_learning/<RL_LIBRARY>/play.py \
    --task=<TASK_NAME>-Play \
    --checkpoint=/absolute/path/to/checkpoint.pth
```

### Analysis
```bash
# Analyze training logs
python scripts/analysis/analyze_logs.py
```

## Environment Variables

You can set environment variables to customize behavior:

```bash
# Linux
export ISAACLAB_PATH=~/IsaacLab
export CUDA_VISIBLE_DEVICES=0

# Windows
set ISAACLAB_PATH=C:\IsaacLab
set CUDA_VISIBLE_DEVICES=0
```

## Next Steps

- Learn about [Project Structure](STRUCTURE.md)
- Read the [Training Guide](TRAINING.md) for detailed training instructions
- Explore [Scripts Documentation](../scripts/README.md)

