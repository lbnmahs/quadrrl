from isaaclab.utils import configclass

from quadrrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from quadrrl.robots.unitree import UNITREE_B2_CFG  # isort: skip


@configclass
class UnitreeB2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-terrain velocity-tracking task for Unitree B2 (QUADRRL style)."""

    def __post_init__(self):
        super().__post_init__()

        # Scene / robot
        self.scene.robot = UNITREE_B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Action scaling: slightly smaller than default, similar to Go2 but with stronger legs.
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        }

        # Events: follow Go2 template but allow a bit more randomization for the heavier robot.
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
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

        # Rewards: start from default QUADRRL velocity task but bias towards stronger tracking
        # and slightly higher torque regularization, inspired by robot_lab's B2 config.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Feet-specific shaping: use generic foot regex and keep modest weights.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.05
        # Penalize contacts on thigh links (match names like FL_thigh, FR_thigh, etc.).
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"

        # Terminations: treat base collisions as failure.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class UnitreeB2RoughEnvCfg_PLAY(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller number of envs and reduced terrain complexity for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable noise & strong perturbations for evaluation.
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

