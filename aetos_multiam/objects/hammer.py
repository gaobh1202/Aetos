"""Cross-peen hammer -- the object that gets handed over in ``AerialHandover``.

NVIDIA SimReady asset (``handTool_hammer_crossPeen_b01``). It ships with
per-engine physics variants and arrives with ``PhysX = Enabled``, so unlike
this repo's other props it needs no fixup: one rigid body, authored mass, and
colliders that come in through the PhysX payload rather than as sibling prims
(nothing shows up under a plain stage traverse, which is expected).

268 x 104 x 36 mm and **627 g**. That is a heavy payload for a SO-ARM100 on a
4 kg drone -- real servos would not hold it at that moment arm -- but in sim
with ``arm_dynamics_residual`` doing its job it is carryable. The handle is
the part a gripper closes on, well inside the jaw's 6.9-142 mm range
(``debug/measure_gripper_range.py``).
"""

from pathlib import Path

from aetos_multiam.utils import make_non_instanceable

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "handTool_hammer_crossPeen_b01" / "simready_usd"
               / "sm_handTool_hammer_crossPeen_b01_01.usd")

#: Overall size in metres, measured off the asset.
SIZE_M = (0.2676, 0.1043, 0.0359)
MASS_KG = 0.6268


class Hammer(AetosObject):
    usd_path = USD_PATH
    root_body_name = "Geometry/hammer_01_obj_00"

    def __init__(self, name: str = "hammer", translation=(0.0, 0.0, 0.0),
                 yaw_deg: float = 0.0, scale: float = 1.0):
        super().__init__(name, translation=translation, yaw_deg=yaw_deg, scale=scale)

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        # The body prim is authored instanceable, with its collider mesh in
        # the prototype -- PhysX then builds no actor at all. See
        # utils/usd_fixups.make_non_instanceable.
        make_non_instanceable(prim_path)
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        # The RigidBodyAPI prim sits two levels under the reference wrapper;
        # a view on the wrapper matches zero rigid bodies.
        super().initialize(f"{prim_paths_expr}/{self.root_body_name}")

    def resting_translation(self, x: float, y: float, surface_z: float):
        """Prim translation that lays the hammer on a surface at ``surface_z``.

        The asset's origin is at the bottom of its bounding box, so this is
        just a nudge clear of the surface rather than a half-height offset.
        """
        return (x, y, surface_z + 0.002)
