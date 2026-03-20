from .rough_env_cfg import DeeproboticsM20RoughEnvCfg

from isaaclab.utils import configclass

@configclass
class DeeproboticsM20FlatEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Flat-terrain variant of M20 task."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "DeeproboticsM20FlatEnvCfg":
            self.disable_zero_weight_rewards()
