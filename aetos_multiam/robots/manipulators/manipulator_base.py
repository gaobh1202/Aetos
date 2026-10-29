"""Manipulator spec: the data-only description of one arm model (robosuite ManipulatorModel analog).

Where robosuite's ``ManipulatorModel`` knows how to merge a ``GripperModel``
onto its own MJCF eef site, a ``ManipulatorSpec`` just records *where* that
mount point is (``eef_body_name``) and whether a gripper is already baked
into its USD (``gripper_bundled``) -- the actual merge happens at spawn time
via ``aetos_multiam.utils.mount_child``, driven by ``AerialManipulatorBase``.

Some manipulators (AM-bench's own arms) are instead baked into the *drone's*
USD at authoring time rather than mounted onto it at runtime -- for those,
``bundled_into_drone=True`` tells ``AerialManipulatorBase.spawn()`` to skip
``mount_child`` entirely; the spec still exists so arm-joint-name-based
introspection (building controllers, reading observations) works the same
way regardless of whether the arm got there via mounting or was pre-baked.
"""

import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from aetos_multiam.assets import AssetNotAvailableError


@dataclass(frozen=True)
class ManipulatorSpec:
    name: str
    usd_path: str
    root_body_name: str
    """Name of the arm's own root rigid body -- what the drone's fixed joint welds to."""
    eef_body_name: str
    """Body a gripper mounts onto, if ``gripper_bundled`` is False."""
    arm_joint_names: Sequence[str] = field(default_factory=tuple)
    gripper_bundled: bool = False
    """True if the gripper is already part of this USD (e.g. airbot_play_gripper)."""
    bundled_into_drone: bool = False
    """True if this arm is pre-baked into the drone's own USD (e.g. AM-bench's
    quad_scorpion.usd/hexa_scorpion.usd) -- spawn() skips mount_child for it."""
    default_gripper: Optional[str] = None
    """Key into GRIPPER_REGISTRY to use when the caller doesn't ask for a specific one."""
    mount_translation: Sequence[float] = (0.0, 0.0, 0.0)
    mount_orientation: Optional[Sequence[float]] = None
    note: str = ""

    def check_asset(self) -> None:
        if self.bundled_into_drone:
            return  # already covered by the drone spec's own check_asset()
        if not osp.isfile(self.usd_path):
            raise AssetNotAvailableError("manipulator:" + self.name, self.usd_path, self.note)


MANIPULATOR_REGISTRY: Dict[str, ManipulatorSpec] = {}


def register_manipulator(spec: ManipulatorSpec) -> ManipulatorSpec:
    if spec.name in MANIPULATOR_REGISTRY:
        raise ValueError(f"Manipulator '{spec.name}' is already registered.")
    MANIPULATOR_REGISTRY[spec.name] = spec
    return spec


def get_manipulator_spec(name: str) -> ManipulatorSpec:
    try:
        return MANIPULATOR_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown manipulator type '{name}'. Available: {sorted(MANIPULATOR_REGISTRY)}"
        ) from None
