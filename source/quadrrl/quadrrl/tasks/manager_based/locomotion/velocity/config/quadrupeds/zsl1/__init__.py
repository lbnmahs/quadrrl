import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Quadrrl-Velocity-Flat-Zsibot-ZSL1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ZsibotZSL1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Quadrrl-Velocity-Rough-Zsibot-ZSL1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ZsibotZSL1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1RoughPPORunnerCfg",
    },
)