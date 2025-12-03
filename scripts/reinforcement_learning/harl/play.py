# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse

# import numpy as np
import sys
import torch

from isaaclab.app import AppLauncher

import os
import numpy as np

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "haddpg",
        "hatd3",
        "hasac",
        "had3qn",
        "maddpg",
        "matd3",
        "mappo",
    ],
    help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    from .HARL.harl.runners import RUNNER_REGISTRY
except Exception:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    harl_root = os.path.join(current_dir, "HARL")
    if harl_root not in sys.path:
        sys.path.insert(0, harl_root)
    from HARL.harl.runners import RUNNER_REGISTRY

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
import isaaclab_tasks  # noqa: F401

import quadrrl.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    args = args_cli.__dict__

    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    algo_args["render"]["use_render"] = True
    algo_args["train"]["model_dir"] = args["dir"]

    env_args = {}
    num_envs = args["num_envs"] if args["num_envs"] is not None else getattr(env_cfg.scene, "num_envs", 1) or 1
    env_cfg.scene.num_envs = num_envs
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["video_settings"] = {}
    env_args["video_settings"]["video"] = False

    # create runner
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)

    obs, _, _ = runner.envs.reset()
    # normalize to (num_envs, num_agents, obs_dim)
    if obs.ndim == 2:
        obs = obs[None, :, :]

    max_action_space = 0

    for space in runner.envs.action_space:
        if space.shape[0] > max_action_space:
            max_action_space = space.shape[0]

    num_envs_runtime = getattr(runner, "env_num", obs.shape[0])
    actions = torch.zeros((num_envs_runtime, runner.num_agents, max_action_space), dtype=torch.float32, device="cuda:0")
    rnn_states = torch.zeros(
        (
            num_envs_runtime,
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=torch.float32,
        device="cuda:0",
    )
    masks = torch.ones(
        (num_envs_runtime, runner.num_agents, 1),
        dtype=torch.float32,
    )

    total_rewards = torch.zeros((num_envs_runtime, runner.num_agents, 1), dtype=torch.float32, device="cuda:0")

    while simulation_app.is_running():
        with torch.inference_mode():
            for agent_id in range(runner.num_agents):
                action, _, rnn_state = runner.actor[agent_id].get_actions(
                    obs[:, agent_id, :], rnn_states[:, agent_id, :], masks[:, agent_id, :], None, None
                )
                action_space = action.shape[1]
                actions[:, agent_id, :action_space] = action
                rnn_states[:, agent_id, :] = rnn_state

            # adapter expects per-env action shape (n_agents, act_dim) for single env
            step_actions = actions if actions.shape[0] > 1 else actions[0]
            obs, _, rewards, dones, _, _ = runner.envs.step(step_actions)
            if obs is not None and obs.ndim == 2:
                obs = np.expand_dims(obs, axis=0)

            # ensure rewards/dones are tensors on the same device for accumulation and masking
            rewards_t = torch.from_numpy(rewards).to(device=total_rewards.device, dtype=total_rewards.dtype) if isinstance(rewards, np.ndarray) else rewards.to(device=total_rewards.device, dtype=total_rewards.dtype)
            total_rewards += rewards_t

            # Suppress per-step average reward printing during play

            dones_t = torch.from_numpy(dones) if isinstance(dones, np.ndarray) else dones
            dones_t = dones_t.to(device=total_rewards.device).to(dtype=torch.bool)
            if dones_t.dim() == 2:
                # shape: (n_envs, n_agents) -> per-env done
                dones_env = torch.all(dones_t, dim=1)
            elif dones_t.dim() == 1:
                # shape: (n_agents,) in single-env adapter; reduce to (1,)
                dones_env = torch.all(dones_t).unsqueeze(0).to(device=total_rewards.device)
            else:
                # Fallback: assume not done
                dones_env = torch.zeros((num_envs_runtime,), dtype=torch.bool, device=total_rewards.device)

            masks = torch.ones((num_envs_runtime, runner.num_agents, 1), dtype=torch.float32, device="cuda:0")
            masks[dones_env] = 0.0
            rnn_states[dones_env] = torch.zeros(
                ((dones_env).sum(), runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
                dtype=torch.float32,
                device="cuda:0",
            )

    runner.envs.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
