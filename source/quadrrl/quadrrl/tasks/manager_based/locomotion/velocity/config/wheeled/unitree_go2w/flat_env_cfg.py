from .rough_env_cfg import UnitreeGo2WRoughEnvCfg

from isaaclab.utils import configclass

@configclass
class UnitreeGo2WFlatEnvCfg(UnitreeGo2WRoughEnvCfg):
    """Flat-terrain variant of Go2W task."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "UnitreeGo2WFlatEnvCfg":
            self.disable_zero_weight_rewards()