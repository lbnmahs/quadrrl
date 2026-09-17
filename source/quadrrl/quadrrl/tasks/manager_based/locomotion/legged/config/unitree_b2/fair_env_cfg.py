# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Fair-Morph-v2 B2 envs: matched tracking + softer rough terrain curriculum."""

from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeB2FlatEnvCfg
from .rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2RoughFairEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Explicit Fair tracking scale (matches native B2).
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Soften early rough curriculum vs native Go2-scaled B2 rough.
        # Native: boxes (0.025, 0.1), random_rough (0.01, 0.06), max_init=5.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.05)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.03)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.005
        self.scene.terrain.max_init_terrain_level = 3
        # Keep action scale 0.25 (native) unless soft terrain alone is insufficient.


@configclass
class UnitreeB2FlatFairEnvCfg(UnitreeB2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
