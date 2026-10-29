"""Spawn-time physics property configs.

Re-exported from ``omni_drones.robots.config`` rather than duplicated --
these are generic PhysX rigid-body/articulation knobs, not benchmark logic
(see the "Dependency on OmniDrones" note in docs/ARCHITECTURE.md).
"""

from omni_drones.robots.config import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
    RobotCfg,
)

__all__ = [
    "ArticulationRootPropertiesCfg",
    "RigidBodyPropertiesCfg",
    "RobotCfg",
]
