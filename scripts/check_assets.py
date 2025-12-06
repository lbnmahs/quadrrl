#!/usr/bin/env python3
# Copyright (c) 2024-2025, Laban Njoroge Mahihu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to check if required assets are available and provide guidance."""

import os
import sys

# Try to import Isaac Lab utilities
try:
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab.utils.assets import retrieve_file_path
except ImportError:
    print("ERROR: Cannot import Isaac Lab. Make sure you're using the correct Python environment.")
    print("Try running with: ~/IsaacLab/isaaclab.sh -p scripts/check_assets.py")
    sys.exit(1)

def check_asset(asset_path, asset_name):
    """Check if an asset exists."""
    print(f"\nChecking {asset_name}...")
    print(f"  Path: {asset_path}")
    
    if asset_path.startswith("http"):
        print(f"  Status: URL path detected (requires Omniverse Nucleus connection)")
        print(f"  Action: Assets need to be downloaded from Omniverse Nucleus")
        return False
    elif os.path.exists(asset_path):
        print(f"  Status: ✓ Found locally")
        return True
    else:
        print(f"  Status: ✗ Not found locally")
        print(f"  Action: Asset needs to be downloaded or ISAACLAB_NUCLEUS_DIR needs to be configured")
        return False

def main():
    print("=" * 70)
    print("Isaac Lab Asset Checker")
    print("=" * 70)
    
    print(f"\nISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")
    
    # Check Unitree Go2 USD file
    go2_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
    go2_exists = check_asset(go2_usd_path, "Unitree Go2 USD file")
    
    # Check Anymal C USD file (for comparison)
    anymal_c_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    anymal_c_exists = check_asset(anymal_c_path, "Anymal C USD file")
    
    print("\n" + "=" * 70)
    print("Summary and Recommendations")
    print("=" * 70)
    
    if not go2_exists:
        print("\n❌ Unitree Go2 assets are missing!")
        print("\nTo fix this issue, you have several options:")
        print("\n1. Download assets from Omniverse Nucleus:")
        print("   - Connect to Omniverse Nucleus server")
        print("   - Download the required assets")
        print("   - Set ISAACLAB_NUCLEUS_DIR to point to the local assets directory")
        print("\n2. Set ISAACLAB_NUCLEUS_DIR environment variable:")
        print("   export ISAACLAB_NUCLEUS_DIR=/path/to/local/assets")
        print("\n3. Use Isaac Lab's asset downloader (if available)")
        print("\n4. Check Isaac Lab documentation for asset setup instructions")
        print("   https://isaac-sim.github.io/IsaacLab/")
    else:
        print("\n✓ All required assets are available!")
    
    if ISAACLAB_NUCLEUS_DIR.startswith("http"):
        print("\n⚠️  ISAACLAB_NUCLEUS_DIR is set to a URL.")
        print("   This requires an active Omniverse Nucleus connection.")
        print("   Consider downloading assets locally for offline use.")

if __name__ == "__main__":
    main()

