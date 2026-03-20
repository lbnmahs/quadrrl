# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def _zeros_like_env(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return a zero reward tensor for every environment instance."""
    return torch.zeros(env.num_envs, device=env.device)


def stand_still(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compatibility alias for older configs."""
    return stand_still_joint_deviation_l1(env, command_name, command_threshold, asset_cfg)


def joint_pos_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Compatibility penalty: stronger joint-deviation penalty when standing still."""
    command = env.command_manager.get_command(command_name)
    is_standing = torch.norm(command[:, :2], dim=1) < command_threshold
    base_penalty = mdp.joint_deviation_l1(env, asset_cfg)
    scale = torch.where(is_standing, torch.full_like(base_penalty, stand_still_scale), torch.ones_like(base_penalty))
    return base_penalty * scale


def wheel_vel_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Compatibility fallback for wheel velocity penalty."""
    del sensor_cfg, command_name, velocity_threshold, command_threshold
    return mdp.joint_vel_l2(env, asset_cfg)


def joint_mirror(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mirror_joints: list[list[str]] | None = None,
) -> torch.Tensor:
    """Compatibility fallback for mirror-joint symmetry penalty."""
    del mirror_joints
    return mdp.joint_deviation_l1(env, asset_cfg)


def action_mirror(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mirror_joints: list[list[str]] | None = None,
) -> torch.Tensor:
    """Compatibility fallback for action mirror penalty."""
    del asset_cfg, mirror_joints
    return mdp.action_rate_l2(env)


def action_sync(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_groups: list[list[str]] | None = None,
) -> torch.Tensor:
    """Compatibility fallback for action synchronization penalty."""
    del asset_cfg, joint_groups
    return mdp.action_rate_l2(env)


def feet_air_time_variance_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize variance in per-foot air-time (higher variance -> higher penalty)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.var(last_air_time, dim=1)


def GaitReward(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    max_err: float,
    velocity_threshold: float,
    command_threshold: float,
    synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Compatibility placeholder for custom gait reward from other codebases."""
    del std, command_name, max_err, velocity_threshold, command_threshold, synced_feet_pair_names, asset_cfg, sensor_cfg
    return _zeros_like_env(env)


def feet_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    command_name: str = "base_velocity",
    expect_contact_num: int = 2,
) -> torch.Tensor:
    """Penalize deviation from expected number of contacts."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    n_contacts = contacts.sum(dim=1).float()
    reward = -torch.abs(n_contacts - float(expect_contact_num))
    command = env.command_manager.get_command(command_name)
    reward *= torch.norm(command[:, :2], dim=1) > 0.1
    return reward


def feet_contact_without_cmd(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Encourage contact when velocity command is near zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    n_contacts = contacts.sum(dim=1).float()
    command = env.command_manager.get_command(command_name)
    is_standing = torch.norm(command[:, :2], dim=1) < 0.1
    return n_contacts * is_standing


def feet_stumble(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Compatibility placeholder for stumble penalty."""
    del sensor_cfg
    return _zeros_like_env(env)


def feet_height(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tanh_mult: float = 2.0,
    target_height: float = 0.05,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Compatibility placeholder for foot-height shaping."""
    del asset_cfg, tanh_mult, target_height, command_name
    return _zeros_like_env(env)


def feet_height_body(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tanh_mult: float = 2.0,
    target_height: float = -0.3,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Compatibility placeholder for body-relative foot-height shaping."""
    del asset_cfg, tanh_mult, target_height, command_name
    return _zeros_like_env(env)


def feet_distance_y_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stance_width: float = 0.3,
) -> torch.Tensor:
    """Compatibility placeholder for lateral foot-distance reward."""
    del std, asset_cfg, stance_width
    return _zeros_like_env(env)


def upward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Compatibility placeholder for upward-orientation reward."""
    return _zeros_like_env(env)
