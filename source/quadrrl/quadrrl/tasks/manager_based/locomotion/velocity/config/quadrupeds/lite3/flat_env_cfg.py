from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3FlatEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Flat-ground variant of the Lite3 velocity-tracking task."""

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain and disabled curriculum.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # Disable height scan on flat ground.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

