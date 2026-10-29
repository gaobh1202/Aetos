"""AetosObject -- thin USD-prim wrapper for task objects (robosuite ``MujocoObject`` analog).

Spawning splices a referenced USD into the template env namespace (same
cloning mechanism the robot uses); ``initialize`` then grabs an
``ArticulationView`` (for objects with internal joints, like a button's
plunger) or a plain ``RigidPrimView`` (for rigid fixtures).
"""

import omni.isaac.core.utils.prims as prim_utils
from omni_drones.views import ArticulationView, RigidPrimView


class AetosObject:
    usd_path: str
    has_joints: bool = False

    def __init__(self, name: str, translation=(0.0, 0.0, 0.0), orientation=None):
        self.name = name
        self.translation = translation
        self.orientation = orientation
        self._view = None

    def spawn(self, template_env_ns: str) -> str:
        prim_path = f"{template_env_ns}/{self.name}"
        prim_utils.create_prim(
            prim_path,
            usd_path=self.usd_path,
            translation=self.translation,
            orientation=self.orientation,
        )
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        if self.has_joints:
            self._view = ArticulationView(prim_paths_expr, reset_xform_properties=False)
        else:
            self._view = RigidPrimView(prim_paths_expr, reset_xform_properties=False)
        self._view.initialize()

    def get_world_pose(self, clone: bool = False):
        """Root prim world pose -- good enough as a coarse position proxy for
        reward shaping; use a dedicated body-level view for anything precise."""
        return self._view.get_world_poses(clone=clone)
