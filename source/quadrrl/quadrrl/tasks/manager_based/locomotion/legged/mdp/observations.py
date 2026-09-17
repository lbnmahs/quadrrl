# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for legged locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.observations import height_scan as _isaaclab_height_scan
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan with non-finite hits replaced.

    Missed rays or exploded robot poses can yield NaN/Inf. Downstream clipping
    does not remove NaNs, and a single NaN observation can NaN the PPO std.
    """
    heights = _isaaclab_height_scan(env, sensor_cfg, offset=offset)
    return torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=-1.0)
