from isaaclab.utils import configclass

from quadrrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from quadrrl.robots.deeprobotics import DEEPROBOTICS_LITE3_CFG  # isort: skip


@configclass
class DeeproboticsLite3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-terrain velocity-tracking task for DeepRobotics Lite3 (QUADRRL style)."""

    def __post_init__(self):
        super().__post_init__()

        # Scene / robot
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/TORSO"

        # Scale terrains a bit milder than ANYmal to reflect the smaller robot.
        if self.scene.terrain.terrain_generator is not None:
            if "boxes" in self.scene.terrain.terrain_generator.sub_terrains:
                self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
            if "random_rough" in self.scene.terrain.terrain_generator.sub_terrains:
                self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.05)

        # Joint action scaling: similar structure to robot_lab Lite3 config, but using position-only actions.
        self.actions.joint_pos.scale = {
            ".*_HipX_joint": 0.2,
            "^(?!.*_HipX_joint).*": 0.3,
        }

        # Events: slightly broaden mass randomization on torso, tune reset pose.
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "TORSO"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "TORSO"
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

        # Rewards: closer to robot_lab Lite3 tuning but mapped onto QUADRRL reward set.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Feet / contact shaping (use upper-case foot links).
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_FOOT"
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = "^(?!.*_FOOT).*"

        # Terminations.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "TORSO"


@configclass
class DeeproboticsLite3RoughEnvCfg_PLAY(DeeproboticsLite3RoughEnvCfg):
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

