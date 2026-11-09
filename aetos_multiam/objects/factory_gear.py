"""IsaacLab's Factory gear set, as task objects.

Four USDs -- a base plate with three shafts, plus a large/medium/small gear
that fit those shafts and mesh with each other. They are what IsaacLab's own
``Factory GearMesh`` task uses. Not shipped with Isaac Sim 4.1 (its
``Props/Factory/`` has only nuts and bolts); they live in the 4.5/5.0 asset
trees and load fine here. ``docs/ASSETS.md`` has the download command.

Two things measured off the assets that every caller needs:

**The geometry is baked at an offset from the body origin.** Every xform in
these files is identity; each gear's mesh sits centred over the shaft it
belongs on (large at x=-30.25 mm, medium at +20.25 mm, small at +50.75 mm,
all with their underside at z=+5 mm). So spawning a gear at exactly the
base's own origin seats it on its shaft for free, and putting one anywhere
else means subtracting ``mesh_offset`` -- ``resting_translation()`` does that.

**SDF collision needs the GPU PhysX pipeline.** The colliders are authored
``physics:approximation = "sdf"``, which is the only approximation that lets
a toothed gear actually go down a shaft; under a CPU pipeline they have no
collision at all and fall through the floor (measured: free-fall to -19.95 m
in 2 s). ``cfg/base/sim_base.yaml`` already runs GPU, but a standalone script
built on ``SimulationContext`` has to ask for it.

Sizes, 1:1 (``metersPerUnit`` is 1.0, so these are metres):

===========  ===============  ==========  ==========
asset        outer dia.       bore dia.   height
===========  ===============  ==========  ==========
base         150 x 75 plate   --          25 (shafts)
large        62 mm            10 mm       25 mm
medium       42 mm            10 mm       25 mm
small        22 mm            10 mm       25 mm
===========  ===============  ==========  ==========

Shafts are 9.56 mm across at x = -30.25 / +20.25 / +50.75 mm, so neighbouring
pairs are 50.5 mm (large+medium) and 30.5 mm apart (medium+small) -- each pair
meshes at that spacing. Every piece is 50 g.
"""

from pathlib import Path
from typing import Sequence, Tuple

import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom, UsdPhysics

from aetos_multiam.utils import strip_articulation_root

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets" / "factory_gears"

#: Shaft centres in the base's own frame, metres.
SHAFT_X = {"large": -0.03025, "medium": 0.02025, "small": 0.05075}
#: Underside of every gear, and the top of the base plate they rest on.
GEAR_BOTTOM_Z = 0.005
PLATE_TOP_Z = 0.005
SHAFT_TOP_Z = 0.025
GEAR_HEIGHT = 0.025
GEAR_TOP_Z = 0.030          # gear's top face, in its own (unscaled) body frame
#: Footprint of the base plate, unscaled. A pedestal under it wants this x scale.
PLATE_SIZE_XY = (0.150, 0.075)


class FactoryGearBase(AetosObject):
    """The plate with three shafts. Carries its own fixed joint to the world,
    so it stays wherever it is spawned without any extra kinematic fixup."""

    usd_path = str(_ASSETS / "factory_gear_base.usd")
    root_body_name = "factory_gear_base_loose"

    def __init__(self, name: str = "gear_base", translation=(0.0, 0.0, 0.0), scale: float = 1.0):
        super().__init__(name, translation=translation, scale=scale)

    def initialize(self, prim_paths_expr: str) -> None:
        super().initialize(f"{prim_paths_expr}/{self.root_body_name}")

    def shaft_world_xy(self, size: str) -> Tuple[float, float]:
        """Where one shaft is, in world coordinates."""
        return (self.translation[0] + self.scale * SHAFT_X[size], self.translation[1])


class FactoryGear(AetosObject):
    """One gear. ``size`` is ``large``/``medium``/``small``."""

    def __init__(self, size: str, name: str = None, translation=(0.0, 0.0, 0.0),
                 scale: float = 1.0):
        if size not in SHAFT_X:
            raise ValueError(f"size must be one of {sorted(SHAFT_X)}, got {size!r}")
        self.size = size
        self.usd_path = str(_ASSETS / f"factory_gear_{size}.usd")
        self.root_body_name = f"factory_gear_{size}"
        super().__init__(name or f"gear_{size}", translation=translation, scale=scale)

    @property
    def mesh_offset(self) -> Tuple[float, float, float]:
        """Geometry centre relative to the body origin (see module docstring)."""
        return (self.scale * SHAFT_X[self.size], 0.0, self.scale * GEAR_BOTTOM_Z)

    def resting_translation(self, x: float, y: float, surface_z: float) -> Tuple[float, float, float]:
        """Prim translation that puts this gear upright at ``(x, y)`` with its
        underside on a surface at ``surface_z``."""
        ox, oy, oz = self.mesh_offset
        return (x - ox, y - oy, surface_z - oz)

    #: A plain cylinder standing on the gear's top face, on its axis, for a
    #: gripper to close on. Its size is **absolute**: scaling the gear up for
    #: visibility would otherwise take the graspable feature out of the jaw's
    #: range, and the whole point is that the two are independent. 30 mm needs
    #: the SO-ARM jaw at about +6.5 deg, against a limit of +100
    #: (``debug/measure_gripper_range.py``), and 80 mm of post is just under
    #: the 85 mm finger length.
    grab_post_diameter_m = 0.030
    grab_post_length_m = 0.080

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        # IsaacLab authors every gear as a one-link *articulation*
        # (ArticulationRootAPI on the body prim), because its own wrapper
        # drives them that way. PhysX then never registers them as free rigid
        # bodies, and a RigidPrimView comes back empty -- "did not match any
        # rigid bodies", with no cooking error anywhere to point at it. The
        # base plate carries no such API, which is why that one works untouched.
        strip_articulation_root(f"{prim_path}/{self.root_body_name}")
        self._add_grab_post(f"{prim_path}/{self.root_body_name}")
        return prim_path

    def _add_grab_post(self, body_path: str) -> None:
        if not self.grab_post_diameter_m:
            return
        # Created as a child of the gear's own rigid body, so it is part of
        # that body rather than a second one -- no joint needed. Its numbers
        # are divided by the spawn scale, which lives on the wrapper prim
        # above and would otherwise scale the post along with the gear.
        s = self.scale
        radius = (self.grab_post_diameter_m / 2.0) / s
        height = self.grab_post_length_m / s
        post = prim_utils.create_prim(
            f"{body_path}/grab_post",
            "Cylinder",
            translation=(SHAFT_X[self.size], 0.0, GEAR_TOP_Z + height / 2.0),
            attributes={"radius": radius, "height": height},
        )
        UsdPhysics.CollisionAPI.Apply(post)
        UsdGeom.Gprim(post).CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.35, 0.12)])

    def seated_translation(self, base_translation: Sequence[float]) -> Tuple[float, float, float]:
        """Prim translation that drops this gear straight onto its own shaft of
        a base spawned at ``base_translation`` -- which is that translation
        exactly, because both assets bake the same offset."""
        return tuple(float(v) for v in base_translation)

    def initialize(self, prim_paths_expr: str) -> None:
        super().initialize(f"{prim_paths_expr}/{self.root_body_name}")
