# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Shared Fair-Morph-v2 RSL-RL PPO runners.

Equal policy capacity and obs normalization for Go2↔Go2W, B2↔B2W, ZSL1↔ZSL1W,
and Lite3↔M20 Fair tasks. Logs use ``*_fair`` experiment names so they do not
collide with native ``*-v0`` runs.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FairMorphPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Matched Fair agent: [512,256,128], obs norm on, 20k iters."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "fair_morph"  # overridden per robot/terrain
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2RoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_go2_rough_fair"


@configclass
class UnitreeGo2FlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_go2_flat_fair"


@configclass
class UnitreeGo2WRoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_go2w_rough_fair"


@configclass
class UnitreeGo2WFlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_go2w_flat_fair"


@configclass
class UnitreeB2RoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_b2_rough_fair"


@configclass
class UnitreeB2FlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_b2_flat_fair"


@configclass
class UnitreeB2WRoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_b2w_rough_fair"


@configclass
class UnitreeB2WFlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "unitree_b2w_flat_fair"


@configclass
class ZsibotZSL1RoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "zsibot_zsl1_rough_fair"


@configclass
class ZsibotZSL1FlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "zsibot_zsl1_flat_fair"


@configclass
class ZsibotZSL1WRoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "zsibot_zsl1w_rough_fair"


@configclass
class ZsibotZSL1WFlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "zsibot_zsl1w_flat_fair"


@configclass
class DeeproboticsLite3RoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "deeprobotics_lite3_rough_fair"


@configclass
class DeeproboticsLite3FlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "deeprobotics_lite3_flat_fair"


@configclass
class DeeproboticsM20RoughFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "deeprobotics_m20_rough_fair"


@configclass
class DeeproboticsM20FlatFairPPORunnerCfg(FairMorphPPORunnerCfg):
    experiment_name = "deeprobotics_m20_flat_fair"
