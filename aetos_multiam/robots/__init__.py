"""Drone/manipulator/gripper registries + the aerial-manipulator composition layer.

``manipulators``/``grippers`` are pure data specs with no Isaac Sim
dependency (importable to list/inspect the catalog without a running
simulator, robosuite ``ALL_GRIPPERS``-style); ``drones``/``config``/
``robot_base`` pull in ``omni_drones`` -> ``isaacsim`` transitively. Those
names are loaded lazily via module ``__getattr__`` so that importing
``aetos_multiam.robots.manipulators`` (or ``.grippers``) never requires
Isaac Sim to be importable -- only actually touching a drone-related name
does.
"""

from . import manipulators  # noqa: F401 (populates MANIPULATOR_REGISTRY, no Isaac Sim needed)
from . import grippers  # noqa: F401 (populates GRIPPER_REGISTRY, no Isaac Sim needed)

from .manipulators import MANIPULATOR_REGISTRY, get_manipulator_spec
from .grippers import GRIPPER_REGISTRY, gripper_factory

_LAZY_NAMES = {
    "ArticulationRootPropertiesCfg", "RigidBodyPropertiesCfg", "RobotCfg",
    "AerialManipulatorBase", "AerialManipulatorCfg", "make_aerial_manipulator",
    "DRONE_REGISTRY", "get_drone_spec", "PRESETS", "get_preset",
}


def __getattr__(name):
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in ("ArticulationRootPropertiesCfg", "RigidBodyPropertiesCfg", "RobotCfg"):
        from . import config
        return getattr(config, name)

    if name in ("AerialManipulatorBase", "AerialManipulatorCfg"):
        from . import robot_base
        return getattr(robot_base, name)

    if name == "make_aerial_manipulator":
        return make_aerial_manipulator

    if name in ("DRONE_REGISTRY", "get_drone_spec"):
        from . import drones
        return getattr(drones, name)

    if name in ("PRESETS", "get_preset"):
        from . import presets
        return getattr(presets, name)


def make_aerial_manipulator(cfg, name: str = None):
    """One-call factory, the aerial-manipulator-specific analog of robosuite's
    ``create_robot`` + ``gripper_factory`` + ``robot_base_factory`` trio."""
    from .robot_base import AerialManipulatorBase

    return AerialManipulatorBase(cfg, name=name)


__all__ = [
    "ArticulationRootPropertiesCfg",
    "RigidBodyPropertiesCfg",
    "RobotCfg",
    "AerialManipulatorBase",
    "AerialManipulatorCfg",
    "make_aerial_manipulator",
    "DRONE_REGISTRY",
    "get_drone_spec",
    "MANIPULATOR_REGISTRY",
    "get_manipulator_spec",
    "GRIPPER_REGISTRY",
    "gripper_factory",
    "PRESETS",
    "get_preset",
]
