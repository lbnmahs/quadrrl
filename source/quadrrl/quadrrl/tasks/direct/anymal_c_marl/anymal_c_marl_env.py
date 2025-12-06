# Copyright (c) 2024-2025, Laban Njoroge Mahihu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis

from .anymal_c_marl_env_cfg import AnymalCMultiAgentFlatEnvCfg


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "sphere2": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "arrow1": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "arrow2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "target": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # Yellow for target
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


class AnymalCMultiAgentBar(DirectMARLEnv):
    cfg: AnymalCMultiAgentFlatEnvCfg

    def __init__(
        self, cfg: AnymalCMultiAgentFlatEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)

        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Target positions (x, y) in world frame, relative to environment origin
        self._target_positions = torch.zeros(self.num_envs, 2, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "target_distance_exp",
                "target_reached",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            _base_id, _ = contact_sensor.find_bodies("base")
            _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies(".*THIGH")
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}
        self.object = RigidObject(self.cfg.cfg_rec_prism)
        self.my_visualizer = define_markers()
        self.scene.rigid_objects["object"] = self.object

        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]
            self.contact_sensors[f"robot_{i}"] = ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)])
            self.scene.sensors[f"robot_{i}"] = self.contact_sensors[f"robot_{i}"]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # We need to process the actions for each scene independently
        self.processed_actions = copy.deepcopy(actions)
        for robot_id, robot in self.robots.items():
            self.actions[robot_id] = actions[robot_id].clone()
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * self.actions[robot_id] + robot.data.default_joint_pos
            )

    def _apply_action(self):
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)

        # Get bar position in world frame (relative to env origin)
        bar_pos_w = self.object.data.body_com_pos_w.squeeze(1)  # (num_envs, 3)
        bar_pos_xy = bar_pos_w[:, :2]  # (num_envs, 2)

        # Compute target position relative to bar (in bar's local frame)
        # Target position in world frame relative to env origin
        target_pos_w_xy = self._target_positions  # (num_envs, 2)
        # Relative position from bar to target
        target_relative_xy = target_pos_w_xy - bar_pos_xy  # (num_envs, 2)

        obs = {}

        for robot_id, robot in self.robots.items():
            obs[robot_id] = torch.cat(
                [
                    tensor
                    for tensor in (
                        robot.data.root_com_lin_vel_b,
                        robot.data.root_com_ang_vel_b,
                        robot.data.projected_gravity_b,
                        self._commands,
                        robot.data.joint_pos - robot.data.default_joint_pos,
                        robot.data.joint_vel,
                        self.actions[robot_id],
                        target_relative_xy,  # Target position relative to bar (x, y)
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        # obs = torch.cat(obs, dim=0)
        # observations = {"policy": obs}
        return obs

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

    def _draw_markers(self, command):
        xy_commands = command.clone()
        z_commands = xy_commands[:, 2].clone()
        xy_commands[:, 2] = 0

        bar_pos = self.object.data.body_com_pos_w.squeeze(1).clone()  # (num_envs, 3)
        target_pos_3d = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_3d[:, :2] = self._target_positions
        target_pos_3d[:, 2] = bar_pos[:, 2]  # Use bar's z height for target marker

        marker_ids = torch.concat(
            [
                0 * torch.zeros(2 * self._commands.shape[0]),  # Bar position markers
                1 * torch.ones(self._commands.shape[0]),  # Command arrow
                2 * torch.ones(self._commands.shape[0]),  # Velocity arrow
                3 * torch.ones(self._commands.shape[0]),  # Yaw rate arrow
                4 * torch.ones(self._commands.shape[0]),  # Target position marker
            ],
            dim=0,
        )

        bar_yaw = self.object.data.root_com_ang_vel_b[:, 2].clone()

        scale1 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale1[:, 0] = torch.abs(z_commands)

        scale2 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale2[:, 0] = torch.abs(bar_yaw)

        offset1 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset1[:, 1] = 0

        offset2 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset2[:, 1] = 0

        _90 = (-3.14 / 2) * torch.ones(self._commands.shape[0]).to(self.device)

        marker_orientations = quat_from_angle_axis(
            torch.concat(
                [
                    torch.zeros(3 * self._commands.shape[0]).to(self.device),
                    torch.sign(z_commands) * _90,
                    torch.sign(bar_yaw) * _90,
                    torch.zeros(self._commands.shape[0]).to(self.device),  # Target marker orientation
                ],
                dim=0,
            ),
            torch.tensor([0.0, 1.0, 0.0], device=self.device),
        )

        marker_scales = torch.concat(
            [
                torch.ones((3 * self._commands.shape[0], 3), device=self.device),
                scale1,
                scale2,
                torch.ones((self._commands.shape[0], 3), device=self.device) * 0.2,  # Target marker size
            ],
            dim=0,
        )

        marker_locations = torch.concat(
            [
                bar_pos,
                bar_pos + xy_commands,
                bar_pos + self.object.data.root_com_lin_vel_b,
                bar_pos + offset1,
                bar_pos + offset2,
                target_pos_3d,  # Target position
            ],
            dim=0,
        )

        self.my_visualizer.visualize(
            marker_locations, marker_orientations, scales=marker_scales, marker_indices=marker_ids
        )

    def _get_rewards(self) -> dict:
        bar_commands = torch.stack([-self._commands[:, 1], self._commands[:, 0], self._commands[:, 2]]).t()

        self._draw_markers(bar_commands)

        # Get bar position in world frame
        bar_pos_w = self.object.data.body_com_pos_w.squeeze(1)  # (num_envs, 3)
        bar_pos_xy = bar_pos_w[:, :2]  # (num_envs, 2)

        # Compute distance to target
        target_error_xy = self._target_positions - bar_pos_xy  # (num_envs, 2)
        target_distance = torch.norm(target_error_xy, dim=1)  # (num_envs,)

        # Primary reward: distance to target (exponential decay)
        target_distance_mapped = torch.exp(-target_distance / self.cfg.target_tolerance)

        # Bonus reward for reaching target
        target_reached = target_distance < self.cfg.target_tolerance
        target_reached_bonus = target_reached.float() * self.cfg.target_reached_reward_scale

        # Secondary rewards: velocity tracking (for smoothness)
        # xy linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(bar_commands[:, :2] - self.object.data.root_com_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error)

        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.object.data.root_com_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error)

        rewards = {
            "target_distance_exp": target_distance_mapped * self.cfg.target_distance_reward_scale * self.step_dt,
            "target_reached": target_reached_bonus * self.step_dt,
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return {"robot_0": reward, "robot_1": reward}

    def _get_anymal_fallen(self):
        agent_dones = []

        for _, robot in self.robots.items():
            died = robot.data.body_com_pos_w[:, 0, 2].view(-1) < self.cfg.anymal_min_z_pos
            agent_dones.append(died)

        return torch.any(torch.stack(agent_dones), dim=0)

    def _get_bar_fallen(self):
        bar_z_pos = self.object.data.body_com_pos_w[:, :, 2].view(-1)
        bar_roll_angle = torch.abs(self.get_y_euler_from_quat(self.object.data.root_com_quat_w))

        bar_angle_maxes = bar_roll_angle > self.cfg.max_bar_roll_angle_rad
        bar_fallen = bar_z_pos < self.cfg.bar_z_min_pos

        return torch.logical_or(bar_angle_maxes, bar_fallen)

    def _get_timeouts(self):
        return self.episode_length_buf >= self.max_episode_length - 1

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = self._get_timeouts()
        anymal_fallen = self._get_anymal_fallen()
        bar_fallen = self._get_bar_fallen()

        dones = torch.logical_or(anymal_fallen, bar_fallen)

        return {key: time_out for key in self.robots.keys()}, {key: dones for key in self.robots.keys()}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] = object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(object_default_state, env_ids)
        self.object.reset(env_ids)

        for agent, _ in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = 0.0
            self.previous_actions[agent][env_ids] = 0.0

        # Sample random velocity commands (policy will learn to use these to reach target)
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Sample target positions relative to bar's starting position
        # Get bar's initial position in world frame
        bar_pos_w = self.object.data.body_com_pos_w.squeeze(1)[env_ids]  # (len(env_ids), 3)
        bar_pos_xy = bar_pos_w[:, :2]  # (len(env_ids), 2)

        # Sample target distance and angle
        target_angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        target_distance = (
            torch.rand(len(env_ids), device=self.device)
            * (self.cfg.target_distance_max - self.cfg.target_distance_min)
            + self.cfg.target_distance_min
        )

        # Compute target position relative to environment origin
        target_offset_x = target_distance * torch.cos(target_angle)
        target_offset_y = target_distance * torch.sin(target_angle)
        self._target_positions[env_ids, 0] = bar_pos_xy[:, 0] + target_offset_x
        self._target_positions[env_ids, 1] = bar_pos_xy[:, 1] + target_offset_y

        for _, robot in self.robots.items():
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = robot._ALL_INDICES
            robot.reset(env_ids)
            if len(env_ids) == self.num_envs:
                # Spread out the resets to avoid spikes in training when many environments reset at a similar time
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

            # Reset robot state
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward_s/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
