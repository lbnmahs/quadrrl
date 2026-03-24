#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Sim-to-sim transfer runner for RSL-RL checkpoints.

This script runs a policy checkpoint trained with one backend (source joint order)
inside a target backend/task by remapping observations and actions according to a
joint-name mapping YAML file.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

# local imports
RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "reinforcement_learning" / "rsl_rl"
sys.path.append(str(RSL_RL_SCRIPT_DIR))
import cli_args  # isort: skip

from packaging import version


@dataclass
class JointTransferMap:
    source_joint_names: list[str]
    target_joint_names: list[str]
    # Index list where each source joint is located in target ordering.
    target_to_source_obs_idx: list[int]
    # Index list where each target joint is located in source ordering.
    source_to_target_action_idx: list[int]

    @classmethod
    def from_lists(cls, source_joint_names: list[str], target_joint_names: list[str]) -> "JointTransferMap":
        if len(source_joint_names) != len(target_joint_names):
            raise ValueError(
                f"Joint count mismatch: source={len(source_joint_names)}, target={len(target_joint_names)}."
            )
        if len(set(source_joint_names)) != len(source_joint_names):
            raise ValueError("source_joint_names contains duplicates.")
        if len(set(target_joint_names)) != len(target_joint_names):
            raise ValueError("target_joint_names contains duplicates.")
        if set(source_joint_names) != set(target_joint_names):
            missing_in_target = sorted(set(source_joint_names) - set(target_joint_names))
            missing_in_source = sorted(set(target_joint_names) - set(source_joint_names))
            raise ValueError(
                "Joint name sets differ between source and target mappings. "
                f"Missing in target: {missing_in_target}. Missing in source: {missing_in_source}."
            )

        source_index = {name: idx for idx, name in enumerate(source_joint_names)}
        target_index = {name: idx for idx, name in enumerate(target_joint_names)}

        return cls(
            source_joint_names=source_joint_names,
            target_joint_names=target_joint_names,
            target_to_source_obs_idx=[target_index[name] for name in source_joint_names],
            source_to_target_action_idx=[source_index[name] for name in target_joint_names],
        )


def _load_mapping(path: str) -> JointTransferMap:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    source_joint_names = data["source_joint_names"]
    target_joint_names = data["target_joint_names"]
    return JointTransferMap.from_lists(source_joint_names, target_joint_names)


def _remap_locomotion_observations(
    observations,
    transfer_map: JointTransferMap,
    *,
    joint_obs_start: int,
    remap_actions_history: bool = True,
):
    """Remap joint-related observation blocks from target order to source order.

    Assumed locomotion layout:
    [base_terms, joint_pos(n), joint_vel(n), last_actions(n), ...]
    """

    if not hasattr(observations, "shape") or len(observations.shape) != 2:
        return observations

    obs = observations.clone()
    num_joints = len(transfer_map.source_joint_names)
    blocks = [joint_obs_start, joint_obs_start + num_joints]
    if remap_actions_history:
        blocks.append(joint_obs_start + 2 * num_joints)

    target_to_source = observations.new_tensor(transfer_map.target_to_source_obs_idx, dtype=torch.long)
    for block_start in blocks:
        block_end = block_start + num_joints
        if block_end > obs.shape[1]:
            continue
        obs[:, block_start:block_end] = obs[:, block_start:block_end].index_select(dim=1, index=target_to_source)
    return obs


def _remap_actions(actions, transfer_map: JointTransferMap):
    if not hasattr(actions, "shape") or len(actions.shape) != 2:
        return actions
    if actions.shape[1] != len(transfer_map.source_joint_names):
        raise ValueError(
            "Action dimension does not match mapping. "
            f"actions={actions.shape[1]}, mapped joints={len(transfer_map.source_joint_names)}"
        )
    source_to_target = actions.new_tensor(transfer_map.source_to_target_action_idx, dtype=torch.long)
    return actions.index_select(dim=1, index=source_to_target)


# add argparse arguments
parser = argparse.ArgumentParser(description="Run sim-to-sim transfer with an RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Target task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--policy_transfer_file", type=str, required=True, help="Joint mapping YAML path.")
parser.add_argument(
    "--joint_obs_start",
    type=int,
    default=12,
    help="Observation index where joint_pos block starts (default assumes locomotion layout).",
)
parser.add_argument("--skip_obs_remap", action="store_true", default=False, help="Skip observation remapping.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""
import importlib.metadata as metadata  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402

try:
    import quadrrl  # noqa: F401, E402
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    quadrrl_src_root = repo_root / "source" / "quadrrl"
    sys.path.insert(0, str(quadrrl_src_root))
    import quadrrl  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    transfer_map = _load_mapping(args_cli.policy_transfer_file)
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for sim-to-sim transfer.")
    print("[INFO] Loaded transfer mapping:")
    print_dict(
        {
            "source_joint_names": transfer_map.source_joint_names,
            "target_joint_names": transfer_map.target_joint_names,
            "checkpoint": args_cli.checkpoint,
            "task": args_cli.task,
        },
        nesting=4,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading source model from: {checkpoint}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "actor_critic", None)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            policy_obs = obs
            if not args_cli.skip_obs_remap:
                policy_obs = _remap_locomotion_observations(
                    obs,
                    transfer_map,
                    joint_obs_start=args_cli.joint_obs_start,
                    remap_actions_history=True,
                )
            source_actions = policy(policy_obs)
            target_actions = _remap_actions(source_actions, transfer_map)
            obs, _, dones, _ = env.step(target_actions)

            if hasattr(policy, "reset"):
                policy.reset(dones)
            elif policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
