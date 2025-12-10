# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain locomotion configuration for Spot MARL.

This builds on the flat Spot MARL configuration and uses rough terrain
for consistent training and demoing across all robots.
"""

from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .flat_env_cfg import SpotMarlFlatEnvCfg

##
# Pre-defined configs
##
from quadrrl.robots.spot import SPOT_CFG  # isort: skip


@configclass
class SpotMarlRoughEnvCfg(SpotMarlFlatEnvCfg):
    """Multi-agent Spot velocity-tracking on rough cobblestone / random terrain."""

    def __post_init__(self):
        # Post init of parent (sets base scene / curriculum behaviour)
        super().__post_init__()

        # Switch robot to Spot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Rough terrain generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 5

        # Ensure contact forces sensor (used by several Spot reward terms) ticks at physics rate
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Height scanner for rough terrain
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"


@configclass
class SpotMarlRoughEnvCfg_PLAY(SpotMarlRoughEnvCfg):
    """Smaller rough-terrain env for rollouts / play."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play & visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable observation corruption for play
        self.observations.agentFR.enable_corruption = False
        self.observations.agentFL.enable_corruption = False
        self.observations.agentHR.enable_corruption = False
        self.observations.agentHL.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
