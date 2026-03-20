# Installation Guide

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 20 GB+ free space
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11 (64-bit)
- **Isaac Lab**: Installed per [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python**: 3.10 or newer

## Installation Steps

1. **Install Isaac Lab** - Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. **Clone Repository**
   ```bash
   git clone https://github.com/lbnmahs/quadrrl.git
   cd quadrrl
   ```

3. **Activate Isaac Lab Environment**
   ```bash
   # Linux
   conda activate isaaclab
   # or: source ~/IsaacLab/isaaclab.sh -p
   
   # Windows
   conda activate isaaclab
   # or: C:\IsaacLab\isaaclab.bat -p
   ```

4. **Install Quadrrl**
   ```bash
   python -m pip install -e source/quadrrl
   ```

5. **Verify Installation**
   ```bash
   python scripts/list_envs.py
   # Windows (if Isaac Lab not in PATH): isaaclab.bat -p scripts/list_envs.py
   ```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'quadrrl'`**: Activate Isaac Lab environment and reinstall
- **`CUDA out of memory`**: Reduce `--num_envs` (e.g., 1024 instead of 4096)
- **Isaac Lab not found on Windows**: Use `isaaclab.bat -p` instead of `python`

## Next Steps

See [Getting Started Guide](GETTING_STARTED.md) to begin using Quadrrl.

