# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backwards-compatibility shim for legacy imports."""
from quadrrl.tasks.manager_based.locomotion.velocity.config.spot_marl.flat_env_cfg import (
    SpotMarlFlatEnvCfg,
    SpotMarlFlatEnvCfg_PLAY,
)
from quadrrl.tasks.manager_based.locomotion.velocity.config.spot_marl.rough_env_cfg import (
    SpotMarlRoughEnvCfg,
    SpotMarlRoughEnvCfg_PLAY,
)

# Legacy aliases for backwards compatibility
SpotMarlEnvCfg = SpotMarlRoughEnvCfg
SpotMarlEnvCfg_PLAY = SpotMarlRoughEnvCfg_PLAY

__all__ = [
    "SpotMarlFlatEnvCfg",
    "SpotMarlFlatEnvCfg_PLAY",
    "SpotMarlRoughEnvCfg",
    "SpotMarlRoughEnvCfg_PLAY",
    "SpotMarlEnvCfg",  # Legacy alias
    "SpotMarlEnvCfg_PLAY",  # Legacy alias
]
