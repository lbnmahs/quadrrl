#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain locomotion configuration for Spot.

This builds on the flat Spot configuration but increases terrain difficulty
to provide more challenging locomotion tasks (higher roughness, larger
height variations and full curriculum over terrain levels).
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from quadrrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .flat_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
    SpotEventCfg,
    SpotObservationsCfg,
    SpotRewardsCfg,
    SpotTerminationsCfg,
)


SPOT_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=11,
    num_cols=25,
    horizontal_scale=0.1,
    # Higher amplitude heightfield for rougher terrain
    vertical_scale=0.015,
    slope_threshold=0.85,
    # Start slightly above perfectly flat, extend to significantly rough
    difficulty_range=(0.2, 1.8),
    use_cache=False,
    sub_terrains={
        # Keep a small proportion of flat patches for recovery / curriculum
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # Mild roughness
        "random_rough_low": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.03, 0.07), noise_step=0.02, border_width=0.25
        ),
        # Stronger roughness
        "random_rough_high": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6, noise_range=(0.06, 0.12), noise_step=0.02, border_width=0.25
        ),
    },
)


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

        # Increase episode length slightly to allow traversing larger terrains
        self.episode_length_s = 25.0

        # Use a rougher terrain generator with curriculum enabled
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=SPOT_ROUGH_TERRAIN_CFG,
            max_init_terrain_level=SPOT_ROUGH_TERRAIN_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        # Ensure contact forces sensor (used by several Spot reward terms) ticks at physics rate
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Keep height scanner disabled (as in flat env) for simplicity
        self.scene.height_scanner = None


@configclass
class SpotRoughEnvCfg_PLAY(SpotRoughEnvCfg):
    """Smaller rough-terrain env for rollouts / play."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play & visualization
        self.scene.num_envs = 64
        self.scene.env_spacing = 3.0

        # Limit maximum initial terrain level so that robots spawn across a
        # subset of the curriculum rather than the entire difficulty range.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8

        # Disable randomization for deterministic rollouts if desired
        self.observations.policy.enable_corruption = False
