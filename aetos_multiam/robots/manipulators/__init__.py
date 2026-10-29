from .manipulator_base import (
    MANIPULATOR_REGISTRY,
    ManipulatorSpec,
    get_manipulator_spec,
    register_manipulator,
)

from . import airbot_play  # noqa: F401
from . import dynamixel_2r  # noqa: F401
from . import ambench_quad_arm  # noqa: F401
from . import ambench_hexa_arm  # noqa: F401
from . import so_arm100  # noqa: F401

__all__ = [
    "MANIPULATOR_REGISTRY",
    "ManipulatorSpec",
    "get_manipulator_spec",
    "register_manipulator",
]
