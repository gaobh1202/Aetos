"""Aetos_MultiAM: a benchmark for multi aerial-manipulator (AM) collaboration.

One robot *type* -- an aerial manipulator (drone + optional arm + optional
gripper) -- composed from swappable drone / manipulator / gripper models,
built on Isaac Sim's ``omni.isaac.core`` API (no IsaacLab dependency).

See ``docs/ARCHITECTURE.md`` for how this package's layout maps onto its
reference projects (robosuite, dexmimicgen, OmniDrones, AM-bench).
"""

__all__ = ["init_simulation_app"]


def __getattr__(name):
    # Deferred: importing `aetos_multiam` (or any of its data-only spec
    # submodules, e.g. for listing registered manipulators without a running
    # simulator) must not require Isaac Sim's `isaacsim` module to be
    # importable. Only touching `init_simulation_app` itself does.
    if name == "init_simulation_app":
        from omni_drones import init_simulation_app

        return init_simulation_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
