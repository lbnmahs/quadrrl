# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain locomotion configuration for Spot.

This builds on the flat Spot configuration and uses the standard rough terrain
configuration for consistent training and demoing across all robots.
"""

from isaaclab.utils import configclass

from quadrrl.tasks.manager_based.locomotion.legged.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

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
    """Spot velocity-tracking on rough terrain."""

    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.scene.height_scanner = None


@configclass
class SpotRoughEnvCfg_PLAY(SpotRoughEnvCfg):
    """Smaller rough-terrain env for rollouts / play."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
