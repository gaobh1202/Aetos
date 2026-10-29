"""``gripper_factory`` -- robosuite's ``models/grippers/gripper_factory.py`` analog."""

from typing import Optional

from .gripper_base import GRIPPER_REGISTRY, GripperSpec


def gripper_factory(name: Optional[str]) -> Optional[GripperSpec]:
    """Look up a ``GripperSpec`` by name; ``None`` means "no separate gripper"."""
    if name is None:
        return None
    assert name in GRIPPER_REGISTRY, (
        f"Unknown gripper '{name}'. Available: {sorted(GRIPPER_REGISTRY)}"
    )
    return GRIPPER_REGISTRY[name]
