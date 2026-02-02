# Getting Started

## Prerequisites

- Quadrrl installed (see [Installation Guide](INSTALLATION.md))
- Isaac Lab environment activated
- GPU with CUDA support

## Running Scripts

**Linux**: Use `python` (or `isaaclab.sh -p` if Isaac Lab not in PATH)  
**Windows**: Use `isaaclab.bat -p` for all commands

## Quick Examples

**List Environments**
```bash
python scripts/list_envs.py  # Linux
isaaclab.bat -p scripts/list_envs.py  # Windows
```

**Train a Policy**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-v0 \
    --num_envs=4096 \
    --seed=42
```

**Evaluate a Policy**
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Template-Quadrrl-Velocity-Flat-Anymal-C-Play-v0 \
    --checkpoint=/path/to/checkpoint.pth
```

**Run Demos**
```bash
python scripts/demos/quadrupeds.py
python scripts/demos/usd_policy_inference.py
python scripts/demos/il_anymal_d_usd.py
python scripts/demos/il_go2_rough.py
```

## Common Commands

**Training**
```bash
# Single-agent
python scripts/reinforcement_learning/<RL_LIBRARY>/train.py \
    --task=<TASK_NAME> --num_envs=4096 --seed=42

# Multi-agent (HARL)
python scripts/reinforcement_learning/harl/train.py \
    --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 \
    --num_envs=4096 --algorithm=happo --headless
```

**Analysis**
```bash
python scripts/analysis/analyze_logs.py
```

## Next Steps

- [Training Guide](TRAINING.md) - Detailed training instructions
- [Project Structure](STRUCTURE.md) - Code organization
- [Scripts Documentation](../scripts/README.md) - Available scripts

