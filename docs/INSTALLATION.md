# Installation Guide

This guide covers installation of Quadrrl on both Linux and Windows systems.

## Prerequisites

### Compute Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **CPU**: Multi-core processor (8+ cores recommended for large batches)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 20 GB+ free space for Isaac Lab and generated logs

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11 (64-bit)
- **Isaac Lab**: Installed per the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python**: 3.10 or newer (conda, uv, or virtualenv)
- **CUDA**: Match the version required by your Isaac Lab build
- **Git**: For cloning this repository
- Optional: Omniverse Kit / Isaac Sim for UI workflows

## Installation Steps

### Step 1: Install Isaac Lab

Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for your operating system.

**Important**: Ensure Isaac Lab is properly installed and you can activate its environment before proceeding.

### Step 2: Clone Quadrrl Repository

```bash
git clone https://github.com/lbnmahs/quadrrl.git
cd quadrrl
```

### Step 3: Activate Isaac Lab Environment

#### Linux
```bash
# Option 1: Using conda
conda activate isaaclab

# Option 2: Using Isaac Lab script
source ~/IsaacLab/isaaclab.sh -p
```

#### Windows
```cmd
REM Option 1: Using conda
conda activate isaaclab

REM Option 2: Using Isaac Lab script
C:\IsaacLab\isaaclab.bat -p
```

### Step 4: Install Quadrrl

```bash
# Install in editable mode
python -m pip install -e source/quadrrl
```

### Step 5: (Optional) Install Developer Tooling

```bash
pip install pre-commit
pre-commit install
```

## Verification

Verify your installation by listing available environments:

```bash
# Linux
python scripts/list_envs.py

# Windows (if Isaac Lab not in PATH)
isaaclab.bat -p scripts/list_envs.py
```

You should see a table of available Quadrrl environments.

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'quadrrl'`
- **Solution**: Ensure you've activated the Isaac Lab environment and installed Quadrrl with `pip install -e source/quadrrl`

**Issue**: `CUDA out of memory` errors
- **Solution**: Reduce `--num_envs` parameter (e.g., use 1024 instead of 4096)

**Issue**: Isaac Lab not found on Windows
- **Solution**: Use `isaaclab.bat -p` instead of `python` for all commands

**Issue**: Permission errors on Linux
- **Solution**: Ensure you have write permissions in the installation directory

### Getting Help

- Check [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/)
- Review [Getting Started Guide](GETTING_STARTED.md)
- Open an issue on GitHub

## Next Steps

After installation, proceed to the [Getting Started Guide](GETTING_STARTED.md) to begin using Quadrrl.

