from isaaclab.utils import configclass

from .rough_env_cfg import ZsibotZSL1RoughEnvCfg


@configclass
class ZsibotZSL1FlatEnvCfg(ZsibotZSL1RoughEnvCfg):
    """Flat-ground variant of the ZSL1 velocity-tracking task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

