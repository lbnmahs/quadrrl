import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Quadrrl-MARL-Direct-Anymal-C-v0",
    entry_point=f"{__name__}.anymal_c_marl_env:AnymalCMultiAgentBar",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_marl_env_cfg:AnymalCMultiAgentFlatEnvCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

"""To run the training:
python scripts/harl/train.py --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 --headless

To run the evaluation:
python scripts/harl/play.py --task=Template-Quadrrl-MARL-Direct-Anymal-C-v0 --checkpoint=logs/harl/anymal_c_marl/EXPERIMENT_NAME/exported/policy.pt
"""