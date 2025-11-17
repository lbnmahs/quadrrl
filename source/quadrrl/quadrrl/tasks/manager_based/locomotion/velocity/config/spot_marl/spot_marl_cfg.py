# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backwards-compatibility shim for legacy imports."""
from quadrrl.tasks.manager_based.locomotion.velocity.config.spot_marl.spot_marl_env_cfg import (
    SpotMarlEnvCfg,
    SpotMarlEnvCfg_PLAY,
)

__all__ = ["SpotMarlEnvCfg", "SpotMarlEnvCfg_PLAY"]
