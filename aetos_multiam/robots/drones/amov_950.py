"""AmovDrone, 950mm class -- the real, working drone for this first pass.

Physics (rotor thrust/torque allocation, drag, downwash) is ported wholesale
from ``omni_drones.robots.drone.amov_drone.AmovDrone`` by subclassing it and
overriding only the asset paths, so it points at the *bare* drone USD (no
arm payload baked in) -- the manipulator gets mounted at runtime instead via
``aetos_multiam.utils.mount_child`` (see ``docs/ARCHITECTURE.md``).
"""

import os.path as osp

from omni_drones.robots.drone.amov_drone import AmovDrone as _OmniDronesAmovDrone
from omni_drones.robots.robot import ASSET_PATH as _OMNI_DRONES_ASSET_PATH

from .drone_base import DroneSpec, register_drone

USD_PATH = osp.join(_OMNI_DRONES_ASSET_PATH, "usd", "amov_drone.usd")
PARAM_PATH = osp.join(_OMNI_DRONES_ASSET_PATH, "usd", "amov_drone.yaml")


class Amov950Drone(_OmniDronesAmovDrone):
    """Bare 950mm AmovDrone body (no arm welded into the USD)."""

    usd_path: str = USD_PATH
    param_path: str = PARAM_PATH
    hover_target_usd_path: str = USD_PATH


AMOV_950 = register_drone(
    DroneSpec(
        name="amov_950",
        drone_class=Amov950Drone,
        usd_path=USD_PATH,
        param_path=PARAM_PATH,
        mount_body_name="base_link",
        class_size_mm=950,
        note="Real asset, used by the ButtonGrasp demo task.",
    )
)
