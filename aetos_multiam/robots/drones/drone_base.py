"""Drone spec: the data-only description of one drone model (robosuite RobotModel analog).

A ``DroneSpec`` never touches the simulator by itself -- it just pairs a
``drone_class`` (a ``RobotBase`` subclass that knows how to spawn/apply
thrust) with the asset paths and the body name an arm mounts onto. Actually
spawning happens later, from inside an active ``SimulationContext``, via
``AerialManipulatorBase``.
"""

import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, Type

from aetos_multiam.assets import AssetNotAvailableError


@dataclass(frozen=True)
class DroneSpec:
    name: str
    drone_class: Type  # omni_drones.robots.robot.RobotBase subclass
    usd_path: str
    param_path: str
    mount_body_name: str = "base_link"
    """Body on the drone that a manipulator's root gets mounted/welded onto."""
    class_size_mm: int = 0
    """Nominal rotor-to-rotor diameter class, informational only."""
    note: str = ""

    def check_asset(self) -> None:
        if not osp.isfile(self.usd_path):
            raise AssetNotAvailableError("drone:" + self.name, self.usd_path, self.note)
        # MultirotorBase.__init__ reads param_path unconditionally -- a real
        # USD with a missing/placeholder param_path would otherwise fail with
        # a raw FileNotFoundError deep inside the drone class instead of this
        # spec's clean, explanatory error.
        if not osp.isfile(self.param_path):
            raise AssetNotAvailableError("drone:" + self.name, self.param_path, self.note)


DRONE_REGISTRY: Dict[str, DroneSpec] = {}


def register_drone(spec: DroneSpec) -> DroneSpec:
    if spec.name in DRONE_REGISTRY:
        raise ValueError(f"Drone '{spec.name}' is already registered.")
    DRONE_REGISTRY[spec.name] = spec
    return spec


def get_drone_spec(name: str) -> DroneSpec:
    try:
        return DRONE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown drone type '{name}'. Available: {sorted(DRONE_REGISTRY)}"
        ) from None
