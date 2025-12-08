# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Quadrrl-Velocity-Flat-Spot-MARL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpotMarlFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotMarlPPORunnerCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
        "harl_mappo_unshare_cfg_entry_point": f"{agents.__name__}:harl_mappo_unshare_cfg.yaml",
        "harl_hatrpo_cfg_entry_point": f"{agents.__name__}:harl_hatrpo_cfg.yaml",
        "harl_haa2c_cfg_entry_point": f"{agents.__name__}:harl_haa2c_cfg.yaml",
    },
)

gym.register(
    id="Template-Quadrrl-Velocity-Flat-Spot-MARL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpotMarlFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotMarlPPORunnerCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
        "harl_mappo_unshare_cfg_entry_point": f"{agents.__name__}:harl_mappo_unshare_cfg.yaml",
        "harl_hatrpo_cfg_entry_point": f"{agents.__name__}:harl_hatrpo_cfg.yaml",
        "harl_haa2c_cfg_entry_point": f"{agents.__name__}:harl_haa2c_cfg.yaml",
    },
)

gym.register(
    id="Template-Quadrrl-Velocity-Rough-Spot-MARL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SpotMarlRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotMarlPPORunnerCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
        "harl_mappo_unshare_cfg_entry_point": f"{agents.__name__}:harl_mappo_unshare_cfg.yaml",
        "harl_hatrpo_cfg_entry_point": f"{agents.__name__}:harl_hatrpo_cfg.yaml",
        "harl_haa2c_cfg_entry_point": f"{agents.__name__}:harl_haa2c_cfg.yaml",
    },
)

gym.register(
    id="Template-Quadrrl-Velocity-Rough-Spot-MARL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SpotMarlRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotMarlPPORunnerCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
        "harl_mappo_unshare_cfg_entry_point": f"{agents.__name__}:harl_mappo_unshare_cfg.yaml",
        "harl_hatrpo_cfg_entry_point": f"{agents.__name__}:harl_hatrpo_cfg.yaml",
        "harl_haa2c_cfg_entry_point": f"{agents.__name__}:harl_haa2c_cfg.yaml",
    },
)
