"""Two-handled carry box -- the shared payload in ``AerialTransport``.

Built from primitives rather than referenced from a USD: it has to be one
rigid body with two grab features at a known spacing, and nothing in the
asset library is that. A root Xform carries ``RigidBodyAPI``/``MassAPI``; the
body slab and both handles hang off it as child prims with ``CollisionAPI``,
so PhysX sees one actor with several shapes (same layout IsaacLab's gear
assets use, and the same trick ``FactoryGear``'s grab post uses).

Each handle is a U: two short posts rising from the box's top face near a
**short** end, bridged by a horizontal bar. The bar is what a gripper closes
on -- 30 mm across, which the SO-ARM jaw reaches at about +6.5 deg of its
110 deg travel (``debug/measure_gripper_range.py``), the same figure the
gear grab post was sized to.

The box's own frame has x along its long axis, so spawning it unrotated puts
the long edge along world x and the two handles at +-x.
"""

from typing import Sequence, Tuple

import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom, UsdPhysics

from .object_base import AetosObject

BODY_COLOR = (0.30, 0.45, 0.62)
HANDLE_COLOR = (0.85, 0.35, 0.12)


class CarryBox(AetosObject):
    """A box two aerial manipulators pick up by one handle each."""

    #: Outer size of the slab, metres: long axis x, then y, then height.
    body_size_m = (0.90, 0.40, 0.22)
    #: Total mass, split between two arms when carried.
    mass_kg = 1.0

    handle_bar_diameter_m = 0.030
    handle_bar_length_m = 0.26      # along y
    handle_post_diameter_m = 0.024
    handle_clearance_m = 0.12       # bar height above the box's top face
    #: How far in from each short end the handle stands.
    handle_inset_m = 0.03

    def __init__(self, name: str = "carry_box", translation=(0.0, 0.0, 0.0),
                 yaw_deg: float = 0.0):
        super().__init__(name, translation=translation, yaw_deg=yaw_deg)

    # -- geometry the task needs --------------------------------------------

    @property
    def handle_offset_x(self) -> float:
        """Distance from the box centre to either handle bar, along x."""
        return self.body_size_m[0] / 2.0 - self.handle_inset_m

    @property
    def handle_height_m(self) -> float:
        """Height of a handle bar above the surface the box rests on."""
        return self.body_size_m[2] + self.handle_clearance_m

    def handle_local_offsets(self) -> Tuple[Tuple[float, float, float], ...]:
        """Both handle-bar centres relative to the box's own origin, which sits
        at the middle of its bottom face."""
        return tuple(
            (sign * self.handle_offset_x, 0.0, self.handle_height_m) for sign in (-1.0, 1.0)
        )

    def resting_translation(self, x: float, y: float, surface_z: float) -> Tuple[float, float, float]:
        """Prim translation that stands the box on a surface at ``surface_z``."""
        return (x, y, surface_z)

    # -- spawn ---------------------------------------------------------------

    def spawn(self, template_env_ns: str) -> str:
        prim_path = f"{template_env_ns}/{self.name}"
        root = prim_utils.create_prim(
            prim_path, "Xform",
            translation=self.translation,
            orientation=self.orientation,
        )
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr().Set(float(self.mass_kg))

        w, d, h = self.body_size_m
        self._shape(f"{prim_path}/body", "Cube", (0.0, 0.0, h / 2.0),
                    {"size": 1.0}, BODY_COLOR, scale=(w, d, h))
        for i, (hx, hy, hz) in enumerate(self.handle_local_offsets()):
            self._shape(f"{prim_path}/handle_{i}_bar", "Cylinder", (hx, hy, hz),
                        {"radius": self.handle_bar_diameter_m / 2.0,
                         "height": self.handle_bar_length_m},
                        HANDLE_COLOR,
                        # Cylinders are authored along z; lay this one along y.
                        orientation=(0.7071068, -0.7071068, 0.0, 0.0))
            post_h = self.handle_clearance_m
            for sy in (-1, 1):
                self._shape(
                    f"{prim_path}/handle_{i}_post_{'n' if sy < 0 else 'p'}",
                    "Cylinder",
                    (hx, hy + sy * (self.handle_bar_length_m / 2.0 - 0.02), h + post_h / 2.0),
                    {"radius": self.handle_post_diameter_m / 2.0, "height": post_h},
                    HANDLE_COLOR,
                )
        return prim_path

    def _shape(self, path, prim_type, translation, attributes, color,
               scale=None, orientation=None):
        prim = prim_utils.create_prim(
            path, prim_type, translation=translation, orientation=orientation,
            scale=scale, attributes=attributes,
        )
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return prim

    def initialize(self, prim_paths_expr: str) -> None:
        # The root prim IS the rigid body here -- nothing nested to descend into.
        super().initialize(prim_paths_expr)
