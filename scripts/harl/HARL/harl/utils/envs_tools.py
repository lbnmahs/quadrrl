"""Tools for HARL - Isaac Lab integration only."""
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name != "isaaclab":
        raise NotImplementedError(f"Only 'isaaclab' environment is supported, got '{env_name}'")

    def get_env_fn(rank):
        def init_env():
            if env_name == "isaaclab":
                # Minimal adapter to bridge Isaac Lab DirectMARLEnv to HARL's expected interface
                import copy as _copy
                import numpy as _np
                import torch as _torch
                from gym import spaces as _spaces
                import importlib as _importlib
                try:
                    from isaaclab.envs import DirectMARLEnv as _DirectMARLEnv
                except Exception:
                    from isaaclab.envs import DirectMARLEnv as _DirectMARLEnv

                cfg = env_args["config"]
                # Make a per-process env with a single parallel instance
                if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
                    cfg = _copy.deepcopy(cfg)
                    cfg.scene.num_envs = 1
                    # Disable startup events to avoid mismatches with scene entities during adapter construction
                    if hasattr(cfg, "events"):
                        try:
                            cfg.events = None
                        except Exception:
                            pass

                # Resolve the concrete env class from cfg or Gym registry
                _env_class = getattr(cfg, "class_type", None)
                if _env_class is None and "task" in env_args:
                    try:
                        import gymnasium as _gym
                        _spec = _gym.spec(env_args["task"])
                        _entry = _spec.entry_point
                        if isinstance(_entry, str) and ":" in _entry:
                            _module_path, _class_name = _entry.split(":")
                            _module = _importlib.import_module(_module_path)
                            _env_class = getattr(_module, _class_name, None)
                    except Exception:
                        _env_class = None
                if _env_class is None:
                    _env_class = _DirectMARLEnv

                isaac_env = _env_class(cfg)

                class _IsaacLabHARLEnv:
                    def __init__(self, wrapped_env, cfg_obj, _n_threads):
                        self._env = wrapped_env
                        self._n_threads = _n_threads
                        # agent ids order
                        if hasattr(cfg_obj, "possible_agents"):
                            self._agent_ids = list(cfg_obj.possible_agents)
                        elif hasattr(cfg_obj, "action_spaces"):
                            self._agent_ids = list(cfg_obj.action_spaces.keys())
                        else:
                            raise NotImplementedError("Config must define possible_agents or action_spaces")
                        self.n_agents = len(self._agent_ids)
                        # build spaces lists indexed by agent
                        self.observation_space = [
                            _spaces.Box(low=-_np.inf, high=_np.inf, shape=(_obs_dim,), dtype=_np.float32)
                            for _obs_dim in [cfg_obj.observation_spaces[a] for a in self._agent_ids]
                        ]
                        # EP state: share obs equals obs by default
                        self.share_observation_space = [
                            _spaces.Box(low=-_np.inf, high=_np.inf, shape=(_obs_dim,), dtype=_np.float32)
                            for _obs_dim in [cfg_obj.observation_spaces[a] for a in self._agent_ids]
                        ]
                        self.action_space = [
                            _spaces.Box(low=-1.0, high=1.0, shape=(_act_dim,), dtype=_np.float32)
                            for _act_dim in [cfg_obj.action_spaces[a] for a in self._agent_ids]
                        ]

                    def seed(self, seed_value):
                        _torch.manual_seed(seed_value)
                        if hasattr(self._env, "seed"):
                            try:
                                self._env.seed(seed_value)
                            except Exception:
                                pass

                    def reset(self):
                        obs = self._env.reset()
                        # Some Isaac Lab envs return (obs, extras). Use the first element if a tuple.
                        obs_primary = obs[0] if isinstance(obs, tuple) else obs
                        # obs: dict(agent_id -> torch.Tensor [B, obs_dim]) or [obs_dim]
                        per_agent = []
                        for a in self._agent_ids:
                            arr = obs_primary[a].detach().cpu().numpy()
                            arr = arr.reshape(-1)  # (obs_dim,)
                            per_agent.append(arr)
                        per_agent = _np.stack(per_agent, axis=0)  # (n_agents, obs_dim)
                        obs_per_env = _np.ascontiguousarray(per_agent, dtype=_np.float32)  # (n_agents, obs_dim)
                        share_obs_per_env = obs_per_env.copy()
                        available_actions = None
                        return obs_per_env, share_obs_per_env, available_actions

                    def step(self, actions):
                        # actions: (n_agents, act_dim) -> dict agent -> torch[1, act_dim]
                        act_dict = {
                            a: _torch.as_tensor(actions[idx][None, :], device=self._env.device, dtype=_torch.float32)
                            for idx, a in enumerate(self._agent_ids)
                        }
                        obs, rew, terminated, truncated, info = self._env.step(act_dict)
                        # Convert outputs
                        obs_primary = obs[0] if isinstance(obs, tuple) else obs
                        per_agent = []
                        for a in self._agent_ids:
                            arr = obs_primary[a].detach().cpu().numpy()
                            arr = arr.reshape(-1)  # (obs_dim,)
                            per_agent.append(arr)
                        obs_per_env = _np.ascontiguousarray(_np.stack(per_agent, axis=0), dtype=_np.float32)  # (n_agents, obs_dim)
                        share_obs_per_env = obs_per_env.copy()
                        rews = _np.stack(
                            [rew[a].detach().cpu().numpy() for a in self._agent_ids], axis=0
                        )  # (n_agents, 1)
                        dones_per_agent = _np.array(
                            [
                                bool(terminated.get(a, False)) or bool(truncated.get(a, False))
                                for a in self._agent_ids
                            ],
                            dtype=bool,
                        )  # (n_agents,)
                        infos = [{agent_idx: {} for agent_idx in range(self.n_agents)}]
                        available_actions = None
                        return obs_per_env, share_obs_per_env, rews, dones_per_agent, infos, available_actions

                    def close(self):
                        self._env.close()

                    def render(self, mode="human"):
                        if hasattr(self._env, "render"):
                            return self._env.render(mode=mode)
                        return None

                env = _IsaacLabHARLEnv(isaac_env, cfg, n_threads)
            else:
                raise NotImplementedError(f"Environment '{env_name}' is not supported")
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name != "isaaclab":
        raise NotImplementedError(f"Only 'isaaclab' environment is supported, got '{env_name}'")

    def get_env_fn(rank):
        def init_env():
            if env_name == "isaaclab":
                # For evaluation, reuse the same adapter with single env instance
                import copy as _copy
                import numpy as _np
                import torch as _torch
                from gym import spaces as _spaces
                import importlib as _importlib
                try:
                    from isaaclab.envs import DirectMARLEnv as _DirectMARLEnv
                except Exception:
                    from isaaclab.envs import DirectMARLEnv as _DirectMARLEnv

                cfg = env_args["config"]
                if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
                    cfg = _copy.deepcopy(cfg)
                    cfg.scene.num_envs = 1
                    if hasattr(cfg, "events"):
                        try:
                            cfg.events = None
                        except Exception:
                            pass
                # Resolve the concrete env class from cfg or Gym registry
                _env_class = getattr(cfg, "class_type", None)
                if _env_class is None and "task" in env_args:
                    try:
                        import gymnasium as _gym
                        _spec = _gym.spec(env_args["task"])
                        _entry = _spec.entry_point
                        if isinstance(_entry, str) and ":" in _entry:
                            _module_path, _class_name = _entry.split(":")
                            _module = _importlib.import_module(_module_path)
                            _env_class = getattr(_module, _class_name, None)
                    except Exception:
                        _env_class = None
                if _env_class is None:
                    _env_class = _DirectMARLEnv

                isaac_env = _env_class(cfg)

                class _IsaacLabHARLEnv:
                    def __init__(self, wrapped_env, cfg_obj):
                        self._env = wrapped_env
                        if hasattr(cfg_obj, "possible_agents"):
                            self._agent_ids = list(cfg_obj.possible_agents)
                        elif hasattr(cfg_obj, "action_spaces"):
                            self._agent_ids = list(cfg_obj.action_spaces.keys())
                        else:
                            raise NotImplementedError
                        self.n_agents = len(self._agent_ids)
                        self.observation_space = [
                            _spaces.Box(low=-_np.inf, high=_np.inf, shape=(_obs_dim,), dtype=_np.float32)
                            for _obs_dim in [cfg_obj.observation_spaces[a] for a in self._agent_ids]
                        ]
                        self.share_observation_space = [
                            _spaces.Box(low=-_np.inf, high=_np.inf, shape=(_obs_dim,), dtype=_np.float32)
                            for _obs_dim in [cfg_obj.observation_spaces[a] for a in self._agent_ids]
                        ]
                        self.action_space = [
                            _spaces.Box(low=-1.0, high=1.0, shape=(_act_dim,), dtype=_np.float32)
                            for _act_dim in [cfg_obj.action_spaces[a] for a in self._agent_ids]
                        ]

                    def seed(self, seed_value):
                        _torch.manual_seed(seed_value)
                        if hasattr(self._env, "seed"):
                            try:
                                self._env.seed(seed_value)
                            except Exception:
                                pass

                    def reset(self):
                        obs = self._env.reset()
                        obs_primary = obs[0] if isinstance(obs, tuple) else obs
                        obs_stack = _np.stack(
                            [obs_primary[a].detach().cpu().numpy() for a in self._agent_ids], axis=0
                        )
                        obs_stack = _np.transpose(obs_stack, (1, 0, 2))
                        share_obs = obs_stack.copy()
                        available_actions = None
                        return obs_stack, share_obs, available_actions

                    def step(self, actions):
                        act_dict = {
                            a: _torch.as_tensor(actions[idx][None, :], device=self._env.device, dtype=_torch.float32)
                            for idx, a in enumerate(self._agent_ids)
                        }
                        obs, rew, terminated, truncated, info = self._env.step(act_dict)
                        obs_primary = obs[0] if isinstance(obs, tuple) else obs
                        obs_stack = _np.stack(
                            [obs_primary[a].detach().cpu().numpy() for a in self._agent_ids], axis=0
                        )
                        obs_stack = _np.transpose(obs_stack, (1, 0, 2))
                        share_obs = obs_stack.copy()
                        rews = _np.stack(
                            [rew[a].detach().cpu().numpy() for a in self._agent_ids], axis=0
                        )
                        rews = _np.transpose(rews, (1, 0))[:, :, None]
                        dones_per_agent = _np.array(
                            [
                                bool(terminated.get(a, False)) or bool(truncated.get(a, False))
                                for a in self._agent_ids
                            ],
                            dtype=bool,
                        )
                        infos = [{agent_idx: {} for agent_idx in range(self.n_agents)}]
                        available_actions = None
                        return obs_stack, share_obs, rews, dones_per_agent, infos, available_actions

                    def close(self):
                        self._env.close()

                    def render(self, mode="human"):
                        if hasattr(self._env, "render"):
                            return self._env.render(mode=mode)
                        return None

                env = _IsaacLabHARLEnv(isaac_env, cfg)
            else:
                raise NotImplementedError(f"Environment '{env_name}' is not supported")
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    if env_name != "isaaclab":
        raise NotImplementedError(f"Only 'isaaclab' environment is supported, got '{env_name}'")

    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs

    if env_name == "isaaclab":
        # Render with single-instance Isaac Lab env via the same adapter pattern
        import copy as _copy
        import numpy as _np
        import torch as _torch
        from gym import spaces as _spaces
        import importlib as _importlib
        try:
            from isaaclab.envs import DirectMARLEnv as _DirectMARLEnv
        except Exception:
            from isaaclab.envs import DirectMARLEnv as _DirectMARLEnv

        cfg = env_args["config"]
        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg = _copy.deepcopy(cfg)
            # keep requested num_envs for rendering; just disable events
            if hasattr(cfg, "events"):
                try:
                    cfg.events = None
                except Exception:
                    pass

        _env_class = getattr(cfg, "class_type", None)
        if _env_class is None and "task" in env_args:
            try:
                import gymnasium as _gym
                _spec = _gym.spec(env_args["task"])
                _entry = _spec.entry_point
                if isinstance(_entry, str) and ":" in _entry:
                    _module_path, _class_name = _entry.split(":")
                    _module = _importlib.import_module(_module_path)
                    _env_class = getattr(_module, _class_name, None)
            except Exception:
                _env_class = None
        if _env_class is None:
            _env_class = _DirectMARLEnv

        wrapped = _env_class(cfg)

        class _IsaacLabHARLEnvRender:
            def __init__(self, wrapped_env, cfg_obj):
                self._env = wrapped_env
                if hasattr(cfg_obj, "possible_agents"):
                    self._agent_ids = list(cfg_obj.possible_agents)
                elif hasattr(cfg_obj, "action_spaces"):
                    self._agent_ids = list(cfg_obj.action_spaces.keys())
                else:
                    raise NotImplementedError
                self.n_agents = len(self._agent_ids)
                self.observation_space = [
                    _spaces.Box(low=-_np.inf, high=_np.inf, shape=(cfg_obj.observation_spaces[a],), dtype=_np.float32)
                    for a in self._agent_ids
                ]
                self.share_observation_space = [
                    _spaces.Box(low=-_np.inf, high=_np.inf, shape=(cfg_obj.observation_spaces[a],), dtype=_np.float32)
                    for a in self._agent_ids
                ]
                self.action_space = [
                    _spaces.Box(low=-1.0, high=1.0, shape=(cfg_obj.action_spaces[a],), dtype=_np.float32)
                    for a in self._agent_ids
                ]

            def seed(self, seed_value):
                _torch.manual_seed(seed_value)
                if hasattr(self._env, "seed"):
                    try:
                        self._env.seed(seed_value)
                    except Exception:
                        pass

            def reset(self):
                obs = self._env.reset()
                obs_primary = obs[0] if isinstance(obs, tuple) else obs
                # build (N, A, D)
                per_agent = [_np.asarray(obs_primary[a].detach().cpu().numpy()) for a in self._agent_ids]  # (A lists of (N,D))
                obs_per_env = _np.ascontiguousarray(_np.stack(per_agent, axis=1), dtype=_np.float32)  # (N, A, D)
                share_obs_per_env = obs_per_env.copy()
                available_actions = None
                return obs_per_env, share_obs_per_env, available_actions

            def step(self, actions):
                # actions: (N, A, D) or (A, D)
                act_dict = {}
                for idx, a in enumerate(self._agent_ids):
                    if getattr(actions, "ndim", len(_np.shape(actions))) == 3:
                        a_np = actions[:, idx, :]  # (N, D)
                    else:
                        a_np = actions[idx][None, :]  # (1, D)
                    act_dict[a] = _torch.as_tensor(a_np, device=self._env.device, dtype=_torch.float32)
                obs, rew, terminated, truncated, info = self._env.step(act_dict)
                obs_primary = obs[0] if isinstance(obs, tuple) else obs
                per_agent = [_np.asarray(obs_primary[a].detach().cpu().numpy()) for a in self._agent_ids]  # (A lists of (N,D))
                obs_per_env = _np.ascontiguousarray(_np.stack(per_agent, axis=1), dtype=_np.float32)  # (N, A, D)
                share_obs_per_env = obs_per_env.copy()
                rews = _np.stack(
                    [_np.asarray(rew[a].detach().cpu().numpy()) for a in self._agent_ids],
                    axis=1,
                ).astype(_np.float32)  # (N, A) or (N, A, 1)
                if rews.ndim == 2:
                    rews = rews[..., None]  # ensure shape (N, A, 1)
                # convert terminated/truncated values to CPU numpy (N,) per agent
                n_envs = obs_per_env.shape[0]

                def _to_bool_np(x):
                    if hasattr(x, "detach"):
                        return x.detach().cpu().numpy().astype(_np.bool_)
                    arr = _np.asarray(x)
                    if arr.dtype != _np.bool_:
                        arr = arr.astype(_np.bool_)
                    if arr.shape == ():
                        arr = _np.full((n_envs,), bool(arr), dtype=_np.bool_)
                    return arr
                term = _np.stack([_to_bool_np(terminated.get(a, _np.zeros((n_envs,), dtype=_np.bool_))) for a in self._agent_ids], axis=1)  # (N, A)
                trunc = _np.stack([_to_bool_np(truncated.get(a, _np.zeros((n_envs,), dtype=_np.bool_))) for a in self._agent_ids], axis=1)  # (N, A)
                dones_per_agent = (term | trunc).astype(bool)  # (N, A)
                infos = [{agent_idx: {} for agent_idx in range(self.n_agents)}]
                available_actions = None
                return obs_per_env, share_obs_per_env, rews, dones_per_agent, infos, available_actions

            def close(self):
                self._env.close()

            def render(self, mode="human"):
                if hasattr(self._env, "render"):
                    return self._env.render(mode=mode)
                return None

        env = _IsaacLabHARLEnvRender(wrapped, cfg)
        env_num = getattr(getattr(cfg, "scene", object()), "num_envs", 1) or 1
        manual_render = True
        manual_expand_dims = False if env_num > 1 else True
        manual_delay = True
    else:
        raise NotImplementedError(f"Environment '{env_name}' is not supported")
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "isaaclab":
        return envs.n_agents
    else:
        raise NotImplementedError(f"Environment '{env}' is not supported")
