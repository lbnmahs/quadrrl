# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Fair-Morph-v2 M20 envs: matched tracking weights, upward disabled."""

from isaaclab.utils import configclass

from .flat_env_cfg import DeeproboticsM20FlatEnvCfg
from .rough_env_cfg import DeeproboticsM20RoughEnvCfg


def _apply_fair_wheeled_rewards(cfg) -> None:
    """Match legged tracking scale and drop wheeled-only upward bonus."""
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 0.75
    cfg.rewards.upward.weight = 0.0
    # Parent native cfg only disables zero-weight terms for exact class names;
    # Fair subclasses must re-run after overriding weights.
    cfg.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20RoughFairEnvCfg(DeeproboticsM20RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_fair_wheeled_rewards(self)


@configclass
class DeeproboticsM20FlatFairEnvCfg(DeeproboticsM20FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_fair_wheeled_rewards(self)
