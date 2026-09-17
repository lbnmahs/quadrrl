# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Fair-Morph-v2 Go2 envs: matched tracking weights (already native for legged)."""

from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RoughFairEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Explicit Fair tracking scale (matches native Go2; kept for documentation).
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75


@configclass
class UnitreeGo2FlatFairEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
