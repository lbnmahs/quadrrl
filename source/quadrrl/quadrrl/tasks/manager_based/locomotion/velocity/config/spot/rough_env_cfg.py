#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain locomotion configuration for Spot.

This builds on the flat Spot configuration and uses the standard rough terrain
configuration for consistent training and demoing across all robots.
"""

from isaaclab.utils import configclass

from quadrrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .flat_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
    SpotEventCfg,
    SpotObservationsCfg,
    SpotRewardsCfg,
    SpotTerminationsCfg,
)

##
# Pre-defined configs
##
from quadrrl.robots.spot import SPOT_CFG  # isort: skip


@configclass
class SpotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Spot velocity-tracking on rough cobblestone / random terrain."""

    # MDP / scene components reused from flat configuration
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    def __post_init__(self):
        # Post init of parent (sets base scene / curriculum behaviour)
        super().__post_init__()

        # Switch robot to Spot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Ensure contact forces sensor (used by several Spot reward terms) ticks at physics rate
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Height scanner
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"


@configclass
class SpotRoughEnvCfg_PLAY(SpotRoughEnvCfg):
    """Smaller rough-terrain env for rollouts / play."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play & visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for deterministic rollouts if desired
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
