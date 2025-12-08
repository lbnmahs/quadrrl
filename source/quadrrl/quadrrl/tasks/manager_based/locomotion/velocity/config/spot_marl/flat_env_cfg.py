# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain locomotion configuration for Spot MARL.

This configuration implements multi-agent reinforcement learning where each leg
of the Spot robot is controlled by a separate agent. This allows for better
coordination and specialization compared to single-agent control.
"""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from quadrrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from quadrrl.robots.spot import SPOT_CFG  # isort: skip

# Import shared configs from base
from .spot_marl_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
    SpotEventCfg,
    SpotObservationsCfg,
    SpotRewardsCfg,
    SpotTerminationsCfg,
)


@configclass
class SpotMarlFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Multi-agent Spot velocity-tracking on flat terrain.

    Each leg (FR, FL, HR, HL) is controlled by a separate agent, enabling
    coordinated locomotion through multi-agent reinforcement learning.
    """

    # Basic settings
    class_type = ManagerBasedRLEnv
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # MARL-specific settings
    agents = ["agentFR", "agentFL", "agentHR", "agentHL"]
    possible_agents = ["agentFR", "agentFL", "agentHR", "agentHL"]
    num_actions = None
    num_observations = None
    num_states = None
    action_noise_model = None
    observation_noise_model = None
    action_space = 3
    action_spaces = {agent: 3 for agent in agents}
    # Observation space: 3 (base_lin_vel) + 3 (base_ang_vel) + 3 (projected_gravity) +
    #                    3 (velocity_commands) + 3 (my_joint_pos) + 9 (other_joint_pos) +
    #                    3 (my_joint_vel) + 9 (other_joint_vel) + 3 (my_actions) = 39
    observation_space = 39
    observation_spaces = {agent: 39 for agent in agents}
    state_space = 0
    state_spaces = {agent: 0 for agent in agents}

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        # Note: disable_contact_processing is set for MARL to improve performance
        # with multiple agents. This may affect contact-based rewards slightly.
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class SpotMarlFlatEnvCfg_PLAY(SpotMarlFlatEnvCfg):
    """Smaller flat-terrain env for rollouts / play."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.agentFR.enable_corruption = False
        self.observations.agentFL.enable_corruption = False
        self.observations.agentHR.enable_corruption = False
        self.observations.agentHL.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
