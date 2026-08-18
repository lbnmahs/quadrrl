# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record a video during evaluation (default: enabled).",
)
parser.add_argument(
    "--video_duration",
    type=float,
    default=15.0,
    help="Duration of the recorded video in seconds (used when --video-length is not set).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of the recorded video in environment steps (overrides --video-duration).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, _installed_version: str):
        """Fallback for Isaac Lab versions where deprecation handler was removed."""
        return agent_cfg

from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

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

# PLACEHOLDER: Extension template (do not remove this comment)


def _capture_render_frame(env) -> np.ndarray | None:
    """Capture a single rgb_array frame from an Isaac Lab environment."""
    try:
        frame = env.render(recompute=True)
    except RuntimeError as exc:
        print(f"[WARN] Failed to capture render frame: {exc}")
        return None

    if frame is None:
        return None
    if isinstance(frame, list):
        if len(frame) == 0:
            return None
        frame = frame[-1]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        print(f"[WARN] Unexpected render frame type: {type(frame)}")
        return None
    return frame


def _save_playback_video(frames: list[np.ndarray], video_folder: str, fps: int, name_prefix: str = "rl-video") -> str:
    """Save captured frames to an mp4 file."""
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    os.makedirs(video_folder, exist_ok=True)
    video_path = os.path.join(video_folder, f"{name_prefix}-episode-0.mp4")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(video_path, logger=None)
    del clip
    return video_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_length = None
    render_fps = None
    video_folder = None
    if args_cli.video:
        if args_cli.video_length is not None:
            video_length = args_cli.video_length
        else:
            video_length = max(1, int(round(args_cli.video_duration / env.unwrapped.step_dt)))
        render_fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
        video_folder = os.path.join(log_dir, "videos", "play")
        print(
            f"[INFO] Recording a {args_cli.video_duration:.1f}s video "
            f"({video_length} steps at {env.unwrapped.step_dt:.4f}s/step, {render_fps} fps)."
        )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    video_frames: list[np.ndarray] = []
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            frame = _capture_render_frame(env.unwrapped)
            if frame is not None:
                video_frames.append(frame)
            timestep += 1
            # Exit the play loop after recording one video
            if timestep >= video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.video:
        if video_frames:
            video_path = _save_playback_video(video_frames[:video_length], video_folder, render_fps)
            print(f"[INFO] Saved play video ({len(video_frames[:video_length])} frames) to: {video_path}")
        else:
            print("[WARN] Video recording was enabled but no frames were captured. Check --enable_cameras / render mode.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
