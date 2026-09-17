# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Fair-Morph-v2 ZSL1W envs: matched tracking weights, upward disabled."""

from isaaclab.utils import configclass

from .flat_env_cfg import ZsibotZSL1WFlatEnvCfg
from .rough_env_cfg import ZsibotZSL1WRoughEnvCfg


def _apply_fair_wheeled_rewards(cfg) -> None:
    """Match legged tracking scale and drop wheeled-only upward bonus."""
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 0.75
    cfg.rewards.upward.weight = 0.0
    # Parent native cfg only disables zero-weight terms for exact class names;
    # Fair subclasses must re-run after overriding weights.
    cfg.disable_zero_weight_rewards()


@configclass
class ZsibotZSL1WRoughFairEnvCfg(ZsibotZSL1WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_fair_wheeled_rewards(self)


@configclass
class ZsibotZSL1WFlatFairEnvCfg(ZsibotZSL1WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_fair_wheeled_rewards(self)
