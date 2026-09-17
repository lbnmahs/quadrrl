# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

_FAIR_AGENT = "quadrrl.tasks.manager_based.locomotion.agents.fair_rsl_rl_ppo_cfg"

gym.register(
    id="Quadrrl-Velocity-Flat-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
    },
)

gym.register(
    id="Quadrrl-Velocity-Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    },
)

gym.register(
    id="Quadrrl-Velocity-Flat-Deeprobotics-M20-Fair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fair_env_cfg:DeeproboticsM20FlatFairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_FAIR_AGENT}:DeeproboticsM20FlatFairPPORunnerCfg",
    },
)

gym.register(
    id="Quadrrl-Velocity-Rough-Deeprobotics-M20-Fair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fair_env_cfg:DeeproboticsM20RoughFairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_FAIR_AGENT}:DeeproboticsM20RoughFairPPORunnerCfg",
    },
)
