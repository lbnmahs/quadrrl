# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import math
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from torch.distributions import Normal

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, _installed_version: str):
        """Fallback for Isaac Lab versions where deprecation handler was removed."""
        return agent_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Ensure Quadrrl task configs are registered with Gym before Hydra resolves `--task`.
#
# When running from a source checkout (not installed as a package), we add `source/quadrrl`
# to `sys.path` so `import quadrrl` works.
try:
    import quadrrl  # noqa: F401
except ModuleNotFoundError:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    quadrrl_src_root = repo_root / "source" / "quadrrl"
    sys.path.insert(0, str(quadrrl_src_root))
    import quadrrl  # noqa: F401

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# torch.normal requires std > 0. RSL-RL's unconstrained scalar std (and NaN grads from
# non-finite observations) can violate that during long PPO runs.
_MIN_ACTION_STD = 1e-4


def _sanitize_tensor(value: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf so a single exploded env cannot poison the PPO update."""
    return torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)


def _sanitize_obs(obs):
    """In-place finite cleanup for TensorDict / dict observations."""
    if obs is None:
        return obs
    keys = obs.keys() if hasattr(obs, "keys") else None
    if keys is not None and not torch.is_tensor(obs):
        for key in list(keys):
            obs[key] = _sanitize_obs(obs[key])
        return obs
    if torch.is_tensor(obs):
        return _sanitize_tensor(obs)
    return obs


class FiniteValueRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Drop-in RSL-RL env wrapper that strips non-finite obs/rewards/actions."""

    def get_observations(self):
        return _sanitize_obs(super().get_observations())

    def reset(self):
        obs, extras = super().reset()
        return _sanitize_obs(obs), extras

    def step(self, actions: torch.Tensor):
        if torch.is_tensor(actions):
            actions = torch.nan_to_num(actions, nan=0.0)
        obs, rewards, dones, extras = super().step(actions)
        obs = _sanitize_obs(obs)
        if torch.is_tensor(rewards):
            rewards = _sanitize_tensor(rewards)
        return obs, rewards, dones, extras


def _clamp_policy_action_std(policy, min_std: float = _MIN_ACTION_STD) -> None:
    """Project the learnable action-noise parameter back into a valid range."""
    with torch.no_grad():
        if getattr(policy, "noise_std_type", None) == "scalar" and hasattr(policy, "std"):
            torch.nan_to_num_(policy.std, nan=min_std, posinf=1.0, neginf=min_std)
            policy.std.clamp_(min=min_std)
        elif getattr(policy, "noise_std_type", None) == "log" and hasattr(policy, "log_std"):
            log_min = math.log(min_std)
            torch.nan_to_num_(policy.log_std, nan=0.0, posinf=2.0, neginf=log_min)
            policy.log_std.clamp_(min=log_min)


def _install_positive_action_std_guard() -> None:
    """Keep Gaussian action noise strictly positive in ActorCritic.sample()."""

    def _safe_update_distribution(self, obs):
        mean = torch.nan_to_num(self.actor(obs), nan=0.0, posinf=10.0, neginf=-10.0)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        std = torch.nan_to_num(std, nan=_MIN_ACTION_STD, posinf=1.0, neginf=_MIN_ACTION_STD)
        std = torch.clamp(std, min=_MIN_ACTION_STD)
        self.distribution = Normal(mean, std)

    ActorCritic.update_distribution = _safe_update_distribution
    ActorCriticRecurrent.update_distribution = _safe_update_distribution


_install_positive_action_std_guard()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = FiniteValueRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # Keep the learnable action std valid after every optimizer step. A single
    # non-finite minibatch can otherwise push scalar std to NaN/negative and
    # crash the next `torch.normal` call in the same PPO update.
    if hasattr(runner, "alg") and hasattr(runner.alg, "optimizer"):
        orig_optimizer_step = runner.alg.optimizer.step

        def _step_and_clamp_std(*args, **kwargs):
            result = orig_optimizer_step(*args, **kwargs)
            _clamp_policy_action_std(runner.alg.policy)
            return result

        runner.alg.optimizer.step = _step_and_clamp_std
        _clamp_policy_action_std(runner.alg.policy)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        _clamp_policy_action_std(runner.alg.policy)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
