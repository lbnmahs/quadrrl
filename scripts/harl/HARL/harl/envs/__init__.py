"""Environment logger registry for HARL."""
from harl.envs.isaaclab.isaaclab_logger import IsaacLabLogger

LOGGER_REGISTRY = {
    "isaaclab": IsaacLabLogger,
}
