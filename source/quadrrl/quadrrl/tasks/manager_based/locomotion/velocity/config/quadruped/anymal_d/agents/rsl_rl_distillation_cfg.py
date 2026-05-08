# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)


@configclass
class AnymalDFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 5000
    save_interval = 50
    experiment_name = "anymal_d_flat"
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class AnymalDFlatDistillationRunnerRecurrentCfg(AnymalDFlatDistillationRunnerCfg):
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=True,
    )
