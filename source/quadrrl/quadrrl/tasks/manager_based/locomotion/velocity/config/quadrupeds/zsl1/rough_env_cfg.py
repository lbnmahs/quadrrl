from isaaclab.utils import configclass

from quadrrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from quadrrl.robots.zsibot import ZSIBOT_ZSL1_CFG  # isort: skip


@configclass
class ZsibotZSL1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-terrain velocity-tracking task for Zsibot ZSL1 (QUADRRL style)."""

    def __post_init__(self):
        super().__post_init__()

        # Scene / robot
        self.scene.robot = ZSIBOT_ZSL1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/BASE_LINK"

        # Terrain a bit milder than ANYmal; robot_lab ZSL1 uses moderate heights.
        if self.scene.terrain.terrain_generator is not None:
            if "boxes" in self.scene.terrain.terrain_generator.sub_terrains:
                self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
            if "random_rough" in self.scene.terrain.terrain_generator.sub_terrains:
                self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.05)

        # Actions: treat ABAD joints as slightly stiffer (smaller action range).
        self.actions.joint_pos.scale = {
            ".*_ABAD_JOINT": 0.2,
            "^(?!.*_ABAD_JOINT).*": 0.3,
        }

        # Events
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards (mapped from robot_lab ZSL1, but using QUADRRL terms).
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_FOOT_LINK"
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = "^(?!.*_FOOT_LINK).*"

        # Terminations.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "BASE_LINK"


@configclass
class ZsibotZSL1RoughEnvCfg_PLAY(ZsibotZSL1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

