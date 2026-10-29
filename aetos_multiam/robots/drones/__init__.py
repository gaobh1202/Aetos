from .drone_base import DRONE_REGISTRY, DroneSpec, get_drone_spec, register_drone

# Import side effects populate DRONE_REGISTRY (mirrors dexmimicgen's
# import-everything-in-__init__ registration style).
from . import amov_950  # noqa: F401
from . import omnidrones_340  # noqa: F401
from . import omnidrones_430  # noqa: F401
from . import ambench_600  # noqa: F401
from . import ambench_750  # noqa: F401
from . import iris_650  # noqa: F401

__all__ = ["DRONE_REGISTRY", "DroneSpec", "get_drone_spec", "register_drone"]
