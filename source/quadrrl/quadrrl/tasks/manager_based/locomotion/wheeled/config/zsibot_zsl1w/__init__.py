# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

_FAIR_AGENT = "quadrrl.tasks.manager_based.locomotion.agents.fair_rsl_rl_ppo_cfg"

gym.register(
    id="Quadrrl-Velocity-Flat-Zsibot-ZSL1W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ZsibotZSL1WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1WFlatPPORunnerCfg",
    },
)

gym.register(
    id="Quadrrl-Velocity-Rough-Zsibot-ZSL1W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ZsibotZSL1WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1WRoughPPORunnerCfg",
    },
)

gym.register(
    id="Quadrrl-Velocity-Flat-Zsibot-ZSL1W-Fair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fair_env_cfg:ZsibotZSL1WFlatFairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_FAIR_AGENT}:ZsibotZSL1WFlatFairPPORunnerCfg",
    },
)

gym.register(
    id="Quadrrl-Velocity-Rough-Zsibot-ZSL1W-Fair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fair_env_cfg:ZsibotZSL1WRoughFairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_FAIR_AGENT}:ZsibotZSL1WRoughFairPPORunnerCfg",
    },
)
