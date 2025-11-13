# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the ANYmal-C rough terrain environment.

.. code-block:: bash

    # Usage
    python scripts/interactive_locomotion.py --checkpoint path/to/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import os
import sys

# Import cli_args directly from file to avoid shadowing rsl_rl package
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cli_args_path = os.path.join(scripts_dir, "rsl_rl", "cli_args.py")
spec = importlib.util.spec_from_file_location("cli_args", cli_args_path)
cli_args = importlib.util.module_from_spec(spec)
sys.modules["cli_args"] = cli_args
spec.loader.exec_module(cli_args)

# Ensure local rsl_rl directory doesn't shadow the actual rsl_rl package
# Remove it from sys.modules if it was accidentally registered
if "rsl_rl" in sys.modules:
    local_rsl_rl_path = os.path.join(scripts_dir, "rsl_rl")
    if hasattr(sys.modules["rsl_rl"], "__file__") and sys.modules["rsl_rl"].__file__:
        if sys.modules["rsl_rl"].__file__.startswith(local_rsl_rl_path):
            del sys.modules["rsl_rl"]


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the ANYmal-C rough terrain environment."
)
# append RSL-RL cli arguments (includes --checkpoint argument)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import carb
import omni
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from quadrrl.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg_PLAY

TASK = "Template-Quadrrl-Velocity-Rough-Anymal-C-v0"
RL_LIBRARY = "rsl_rl"


class AnymalCRoughDemo:
    """This class provides an interactive demo for the ANYmal-C rough terrain environment.
    It loads a pre-trained checkpoint for the Template-Quadrrl-Velocity-Rough-Anymal-C-v0 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    The following keyboard commands are available:
    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view
    """

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        # load the trained checkpoint
        if args_cli.checkpoint is not None:
            checkpoint = os.path.abspath(args_cli.checkpoint)
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
        else:
            checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
            if checkpoint is None:
                raise ValueError(
                    f"No published pretrained checkpoint found for {TASK}. "
                    "Please provide a checkpoint path using --checkpoint <path>."
                )
        # create envionrment
        env_cfg = AnymalCRoughEnvCfg_PLAY()
        env_cfg.scene.num_envs = 25
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        
        try:
            # Try to load as JIT model first
            jit_policy = torch.jit.load(checkpoint, map_location=self.device)
            jit_policy.eval()
            
            def jit_policy_wrapper(obs):
                try:
                    policy_obs = obs["policy"]
                    if not isinstance(policy_obs, torch.Tensor):
                        policy_obs = torch.as_tensor(policy_obs, device=self.device)
                    return jit_policy(policy_obs)
                except (KeyError, TypeError, AttributeError):
                    if isinstance(obs, torch.Tensor):
                        return jit_policy(obs)
                    else:
                        if hasattr(obs, "get"):
                            policy_obs = obs.get("policy")
                            if policy_obs is not None:
                                return jit_policy(policy_obs)
                        raise ValueError(
                            f"Could not extract policy observations from {type(obs)}. "
                            "Expected dict/TensorDict with 'policy' key or tensor."
                        )
            self.policy = jit_policy_wrapper
        except Exception:
            # If JIT loading fails, try loading as regular checkpoint
            ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
            ppo_runner.load(checkpoint)
            # obtain the trained policy for inference
            self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            # Escape key exits out of the current selected robot view
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving (only for control keys)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._key_to_control and self._selected_id is not None:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid ANYmal-C robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a ANYmal-C robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    demo_anymal_c = AnymalCRoughDemo()
    obs, _ = demo_anymal_c.env.reset()
    while simulation_app.is_running():
        # check for selected robots
        demo_anymal_c.update_selected_object()
        with torch.inference_mode():
            action = demo_anymal_c.policy(obs)
            obs, _, _, _ = demo_anymal_c.env.step(action)
            # overwrite command based on keyboard input
            # obs is a TensorDict/dict, so access the "policy" key first
            obs["policy"][:, 9:13] = demo_anymal_c.commands


if __name__ == "__main__":
    main()
    simulation_app.close()
