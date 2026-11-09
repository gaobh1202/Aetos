"""Drone spec: the data-only description of one drone model (robosuite RobotModel analog).

A ``DroneSpec`` never touches the simulator by itself -- it just pairs a
``drone_class`` (a ``RobotBase`` subclass that knows how to spawn/apply
thrust) with the asset paths and the body name an arm mounts onto. Actually
spawning happens later, from inside an active ``SimulationContext``, via
``AerialManipulatorBase``.
"""

import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Type

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
    livery: bool = True
    """Whether spawn re-binds this drone's airframe to the shared black-frame/
    red-propeller livery (``utils.apply_drone_livery``). False for the two
    assets that already look right: omnidrones_340, whose own plastic_black/
    plastic_red the livery copies, and ambench_750, whose black propellers
    were kept on request."""
    eef_body_name: Optional[str] = None
    """Only set when this drone's own USD bakes in an arm with no separate
    ManipulatorSpec (e.g. amov_drone_arm): the body a task should track as
    the end effector, directly under the drone's own prim (bodies here are
    flat siblings, not nested under mount_body_name/manipulator/...)."""
    bundled_arm_joint_names: Sequence[str] = field(default_factory=tuple)
    """Only set alongside eef_body_name: static joint-name list for a baked-in
    arm, so tasks can size observation/action specs before initialize() has
    populated the drone's own live arm_joint_indices/num_arm_joints."""
    bundled_gripper_joint_names: Sequence[str] = field(default_factory=tuple)
    """The baked-in arm's own gripper finger joints, if it has any. They are
    NOT driven by a separate gripper controller (there is no GripperSpec for a
    gripper that is already part of the drone's USD) -- the drone class's
    auto-detected "every non-rotor DOF" lumps them in with the arm, so they
    count toward AerialManipulatorBase.num_controlled_arm_joints. Leaving them
    undeclared makes every task's action spec that many dims too short."""
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
