# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the ANYmal-C robot. The robot was trained
using Template-Quadrrl-Velocity-Rough-Anymal-C-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

        # Run the script
        ./isaaclab.sh -p scripts/usd_policy_inference.py --checkpoint logs/rsl_rl/anymal_c_flat/EXPERIMENT_NAME/exported/policy.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an ANYmal-C robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg_PLAY


def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Checkpoint file not found: {policy_path}")
    policy = torch.jit.load(policy_path, map_location=args_cli.device)
    policy.eval()

    # setup environment
    env_cfg = AnymalCFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
