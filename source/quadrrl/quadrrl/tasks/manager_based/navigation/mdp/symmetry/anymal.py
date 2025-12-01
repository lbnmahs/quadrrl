# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal in navigation tasks."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Navigation observations structure:
    - base_lin_vel: [x, y, z] (3D)
    - projected_gravity: [x, y, z] (3D)
    - pose_command: [x, y, heading] (3D)

    Navigation actions:
    - pose_command: [x, y, heading] (3D)

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = obs.repeat(4)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        # -- front-back
        obs_aug["policy"][2 * batch_size : 3 * batch_size] = _transform_policy_obs_front_back(
            env.unwrapped, obs["policy"]
        )
        # -- diagonal
        obs_aug["policy"][3 * batch_size :] = _transform_policy_obs_front_back(
            env.unwrapped, obs_aug["policy"][batch_size : 2 * batch_size]
        )
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(batch_size * 4, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
        # -- front-back
        actions_aug[2 * batch_size : 3 * batch_size] = _transform_actions_front_back(actions)
        # -- diagonal
        actions_aug[3 * batch_size :] = _transform_actions_front_back(actions_aug[batch_size : 2 * batch_size])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    Navigation observation structure:
    - base_lin_vel: [x, y, z] (indices 0:3)
    - projected_gravity: [x, y, z] (indices 3:6)
    - pose_command: [x, y, heading] (indices 6:9)

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # base_lin_vel: [x, y, z] -> [x, -y, z]
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)
    # projected_gravity: [x, y, z] -> [x, -y, z]
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    # pose_command: [x, y, heading] -> [-x, y, -heading]
    # For pose commands: negate x position (left-right flip), keep y, negate heading
    obs[:, 6] = -obs[:, 6]  # x
    obs[:, 7] = obs[:, 7]  # y (unchanged)
    obs[:, 8] = -obs[:, 8]  # heading

    return obs


def _transform_policy_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    Navigation observation structure:
    - base_lin_vel: [x, y, z] (indices 0:3)
    - projected_gravity: [x, y, z] (indices 3:6)
    - pose_command: [x, y, heading] (indices 6:9)

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with front-back symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # base_lin_vel: [x, y, z] -> [-x, y, z]
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, 1], device=device)
    # projected_gravity: [x, y, z] -> [-x, y, z]
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, 1], device=device)
    # pose_command: [x, y, heading] -> [-x, y, pi - heading]
    # For pose commands: negate x position, keep y, transform heading (pi - heading)
    obs[:, 6] = -obs[:, 6]  # x
    obs[:, 7] = obs[:, 7]  # y (unchanged)
    # heading: pi - heading (with wrapping)
    obs[:, 8] = torch.pi - obs[:, 8]

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    Navigation actions are pose commands: [x, y, heading]

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    # pose_command: [x, y, heading] -> [-x, y, -heading]
    actions[:, 0] = -actions[:, 0]  # x
    actions[:, 1] = actions[:, 1]  # y (unchanged)
    actions[:, 2] = -actions[:, 2]  # heading
    return actions


def _transform_actions_front_back(actions: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    Navigation actions are pose commands: [x, y, heading]

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with front-back symmetry applied.
    """
    actions = actions.clone()
    # pose_command: [x, y, heading] -> [-x, y, pi - heading]
    actions[:, 0] = -actions[:, 0]  # x
    actions[:, 1] = actions[:, 1]  # y (unchanged)
    # heading: pi - heading (with wrapping)
    actions[:, 2] = torch.pi - actions[:, 2]
    return actions
