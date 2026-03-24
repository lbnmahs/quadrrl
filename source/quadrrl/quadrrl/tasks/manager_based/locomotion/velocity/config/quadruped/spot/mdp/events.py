# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that can be used to enable Spot randomizations.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _as_torch_tensor(data, device: str | torch.device) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    try:
        import warp as wp  # type: ignore

        if isinstance(data, wp.array):
            return wp.to_torch(data).to(device)
    except Exception:
        pass
    return torch.as_tensor(data, device=device)


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids_long = env_ids.to(dtype=torch.long)
    # get default joint state
    default_joint_pos = _as_torch_tensor(asset.data.default_joint_pos, env.device)
    default_joint_vel = _as_torch_tensor(asset.data.default_joint_vel, env.device)
    soft_joint_pos_limits = _as_torch_tensor(asset.data.soft_joint_pos_limits, env.device)
    soft_joint_vel_limits = _as_torch_tensor(asset.data.soft_joint_vel_limits, env.device)

    joint_min_pos = default_joint_pos[env_ids_long] + position_range[0]
    joint_max_pos = default_joint_pos[env_ids_long] + position_range[1]
    joint_min_vel = default_joint_vel[env_ids_long] + velocity_range[0]
    joint_max_vel = default_joint_vel[env_ids_long] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = soft_joint_pos_limits[env_ids_long, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    # clip vel to range
    joint_vel_abs_limits = soft_joint_vel_limits[env_ids_long]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    # sample these values randomly
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
