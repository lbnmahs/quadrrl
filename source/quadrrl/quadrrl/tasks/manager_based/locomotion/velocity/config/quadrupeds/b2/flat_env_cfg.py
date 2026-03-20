from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2FlatEnvCfg(UnitreeB2RoughEnvCfg):
    """Flat-ground velocity-tracking task for Unitree B2."""

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain and no terrain curriculum.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # No height scan when on flat ground.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None


class UnitreeB2FlatEnvCfg_PLAY(UnitreeB2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller number of envs and reduced terrain complexity for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation corruption for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing event.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
