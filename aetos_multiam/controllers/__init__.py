"""Part controllers for the drone/arm/gripper, plus the composite controller.

``JointPositionController``/``BinaryGripperController``/
``AerialManipulatorController`` are pure ``torch`` math -- importable without
Isaac Sim. ``ControllerBase``/``LeePositionController``/etc. pull in
``omni_drones`` -> ``isaacsim`` transitively; loaded lazily (same pattern as
``aetos_multiam.robots``) so that doesn't gate the pure-math pieces.
"""

from .composite_controller import AerialManipulatorController
from .arm import JointPositionController
from .gripper import BinaryGripperController

_LAZY_NAMES = {"ControllerBase", "LeePositionController", "AttitudeController", "RateController"}


def __getattr__(name):
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "ControllerBase":
        from .controller_base import ControllerBase
        return ControllerBase
    from .drone import AttitudeController, LeePositionController, RateController
    return {"AttitudeController": AttitudeController, "LeePositionController": LeePositionController,
            "RateController": RateController}[name]


__all__ = [
    "ControllerBase",
    "AerialManipulatorController",
    "JointPositionController",
    "BinaryGripperController",
    "LeePositionController",
    "AttitudeController",
    "RateController",
]
