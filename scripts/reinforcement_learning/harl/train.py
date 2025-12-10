# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

import os

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
parser.add_argument("--save_interval", type=int, default=None, help="How often to save the model")
parser.add_argument("--log_interval", type=int, default=None, help="How often to log outputs")
parser.add_argument("--exp_name", type=str, default="test", help="Name of the Experiment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")

parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "mappo",
        "mappo_unshare",
    ],
    help="Algorithm name. Choose from: happo, hatrpo, haa2c, mappo, and mappo_unshare.",
)

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


def _resolve_harl_task_slug(task_name: str | None) -> str | None:
    """Map known HARL task ids to a log directory slug."""
    if not task_name:
        return None
    lowered = task_name.lower()
    if "spot-marl" in lowered:
        return "spot_marl"
    if "anymal" in lowered and "marl" in lowered:
        return "anymal_c_marl"
    return None


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    args = args_cli.__dict__

    args["env"] = "isaaclab"

    args["algo"] = args["algorithm"]

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    # determine number of envs/threads: prefer CLI, else fall back to task config
    default_num_envs = getattr(getattr(env_cfg, "scene", object()), "num_envs", None)
    resolved_num_envs = args["num_envs"] if args["num_envs"] is not None else default_num_envs
    args["num_envs"] = resolved_num_envs
    # Isaac Lab environments manage their own vectorization; keep HARL wrapper single-threaded
    algo_args["train"]["n_rollout_threads"] = 1
    if args["num_env_steps"] is not None:
        algo_args["train"]["num_env_steps"] = args["num_env_steps"]
    if args["save_interval"] is not None:
        algo_args["train"]["eval_interval"] = args["save_interval"]
    if args["log_interval"] is not None:
        algo_args["train"]["log_interval"] = args["log_interval"]
    algo_args["train"]["model_dir"] = args["dir"]
    algo_args["seed"]["specify_seed"] = True
    algo_args["seed"]["seed"] = args["seed"]

    env_args = {}
    if resolved_num_envs is not None and hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = resolved_num_envs
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    env_args["video_settings"] = {
        "video": args_cli.video,
        "video_length": args["video_length"],
        "video_interval": args["video_interval"],
        "log_dir": None,
    }

    task_slug = _resolve_harl_task_slug(args.get("task"))
    if task_slug is not None:
        # Place HARL checkpoints and videos under <repo_root>/logs/harl/<task_slug>/<timestamp>
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        run_root = os.path.join(repo_root, "logs", "harl", task_slug, hms_time)
        os.makedirs(run_root, exist_ok=True)
        if not isinstance(algo_args.get("logger"), dict):
            algo_args["logger"] = {}
        algo_args["logger"]["log_dir"] = run_root
        env_args["video_settings"]["log_dir"] = os.path.join(run_root, "videos")
    else:
        env_args["video_settings"]["log_dir"] = os.path.join(
            algo_args["logger"]["log_dir"],
            "isaaclab",
            args["task"],
            args["algorithm"],
            args["exp_name"],
            "-".join(["seed-{:0>5}".format(agent_cfg["seed"]["seed"]), hms_time]),
            "videos",
        )

    # create runner

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
