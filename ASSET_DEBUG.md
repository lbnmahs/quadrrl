# Asset Debug Guide

## Issue: Missing USD File for Unitree Go2

If you encounter the error:
```
FileNotFoundError: USD file not found at path at: 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'
```

This means the required assets are not available locally and need to be downloaded or configured.

## Solutions

### Option 1: Use Isaac Lab's Asset Downloader (Recommended)

Isaac Lab typically includes tools to download assets. Check the Isaac Lab documentation:
- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

### Option 2: Download from Omniverse Nucleus

1. **Connect to Omniverse Nucleus:**
   - Ensure you have an Omniverse account
   - Connect to the Omniverse Nucleus server
   - The assets should be available at: `omniverse://localhost/NVIDIA/Assets/Isaac/5.1/Isaac/IsaacLab/`

2. **Download the assets:**
   - Navigate to `Robots/Unitree/Go2/`
   - Download `go2.usd` and any dependencies

3. **Set ISAACLAB_NUCLEUS_DIR:**
   ```bash
   export ISAACLAB_NUCLEUS_DIR=/path/to/downloaded/assets
   ```

### Option 3: Check Local Isaac Sim Installation

If you have Isaac Sim installed locally, the assets might be in:
```bash
# Typical locations:
~/.local/share/ov/pkg/isaac-sim-*/exts/omni.isaac.assets/data/
# or
/path/to/isaac-sim/exts/omni.isaac.assets/data/
```

Set the environment variable to point to this location:
```bash
export ISAACLAB_NUCLEUS_DIR=/path/to/isaac-sim/exts/omni.isaac.assets/data
```

### Option 4: Use Isaac Lab Launcher Script

Try running your script using Isaac Lab's launcher, which may handle asset paths better:

```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=Template-Quadrrl-Velocity-Rough-Unitree-Go2-v0 --num_envs 40
```

### Option 5: Check Asset Availability

Run the diagnostic script to check asset availability:

```bash
~/IsaacLab/isaaclab.sh -p scripts/check_assets.py
```

## Verification

After setting up assets, verify they're accessible:

```python
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

go2_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
if os.path.exists(go2_path):
    print(f"✓ Asset found at: {go2_path}")
else:
    print(f"✗ Asset not found at: {go2_path}")
```

## Additional Resources

- Isaac Lab Documentation: https://isaac-sim.github.io/IsaacLab/
- Omniverse Nucleus: https://www.nvidia.com/en-us/omniverse/nucleus/
- Isaac Sim Assets: https://docs.isaacsim.omniverse.nvidia.com/features/environment_setup/assets/

