from __future__ import annotations

from importlib import import_module
from typing import Any


def configure_default_sim_backend(sim_cfg: Any) -> None:
    """Apply backend-safe simulation defaults shared by velocity tasks."""
    physx_cfg = getattr(sim_cfg, "physx", None)
    if physx_cfg is not None and hasattr(physx_cfg, "gpu_max_rigid_patch_count"):
        physx_cfg.gpu_max_rigid_patch_count = 10 * 2**15


def configure_newton_sim(
    sim_cfg: Any,
    *,
    num_substeps: int = 1,
    nefc_per_env: int = 256,
    ls_iterations: int = 10,
    cone: str = "pyramidal",
    ls_parallel: bool = True,
    impratio: float = 1.0,
    debug_mode: bool = False,
) -> None:
    """Configure Newton + MuJoCo-Warp solver on a SimulationCfg instance."""
    NewtonCfg = None
    MJWarpSolverCfg = None
    try:
        newton_cfg_module = import_module("isaaclab.sim._impl.newton_manager_cfg")
        solver_cfg_module = import_module("isaaclab.sim._impl.solvers_cfg")
        NewtonCfg = getattr(newton_cfg_module, "NewtonCfg", None)
        MJWarpSolverCfg = getattr(solver_cfg_module, "MJWarpSolverCfg", None)
    except Exception:
        # Isaac Lab 3.0+ Newton API lives in `isaaclab_newton`.
        pass

    if NewtonCfg is None or MJWarpSolverCfg is None:
        try:
            newton_cfg_module = import_module("isaaclab_newton.physics.newton_manager_cfg")
            NewtonCfg = getattr(newton_cfg_module, "NewtonCfg", None)
            MJWarpSolverCfg = getattr(newton_cfg_module, "MJWarpSolverCfg", None)
        except Exception as exc:
            raise RuntimeError(
                "Newton backend requested but Isaac Lab Newton integration is not importable. "
                "Install Newton dependencies and use the Isaac Lab develop branch with Newton support."
            ) from exc

    if NewtonCfg is None or MJWarpSolverCfg is None:
        raise RuntimeError(
            "Newton backend requested but required classes (NewtonCfg, MJWarpSolverCfg) are unavailable."
        )

    # Isaac Lab API compatibility:
    # - Older Newton integration expects `nefc_per_env`.
    # - Newer Newton integration expects `njmax` and stores backend under `sim.physics`.
    # Keep a higher default budget to avoid frequent "nefc overflow" during locomotion training.
    try:
        solver_cfg = MJWarpSolverCfg(
            nefc_per_env=nefc_per_env,
            ls_iterations=ls_iterations,
            cone=cone,
            ls_parallel=ls_parallel,
            impratio=impratio,
        )
    except TypeError:
        solver_cfg = MJWarpSolverCfg(
            njmax=nefc_per_env,
            ls_iterations=ls_iterations,
            cone=cone,
            ls_parallel=ls_parallel,
            impratio=impratio,
        )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=num_substeps,
        debug_mode=debug_mode,
    )

    if hasattr(sim_cfg, "newton_cfg"):
        sim_cfg.newton_cfg = newton_cfg
    elif hasattr(sim_cfg, "physics"):
        sim_cfg.physics = newton_cfg
    else:
        raise RuntimeError("SimulationCfg has neither 'newton_cfg' nor 'physics' field for backend assignment.")


def get_root_physics_view(asset: Any) -> Any:
    """Resolve a backend-specific root view for articulation/rigid-body property edits."""
    for attr_name in ("root_physx_view", "root_newton_view", "root_view"):
        view = getattr(asset, attr_name, None)
        if view is not None:
            return view
    return None


def require_root_property_api(asset: Any, required_methods: tuple[str, ...]) -> Any:
    """Return a root physics view that supports all required methods."""
    view = get_root_physics_view(asset)
    if view is None:
        raise RuntimeError(
            f"Could not resolve a root physics view for asset '{type(asset).__name__}'. "
            "Expected one of: root_physx_view, root_newton_view, root_view."
        )

    missing_methods = [name for name in required_methods if not hasattr(view, name)]
    if missing_methods:
        raise RuntimeError(
            f"Root physics view '{type(view).__name__}' is missing required methods: {missing_methods}."
        )
    return view
