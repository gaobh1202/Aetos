"""AerialTransport -- two AMs carry one box between them, a handle each.

Scene (``WarehouseArena``, concrete floor, no gantry):

* a **pick table** in the middle of the bay carrying a two-handled box, its
  long axis along world x so the handles sit at -x and +x -- one in front of
  each robot;
* a **drop table** offset along y, empty, with a green rectangle painted on
  it marking where the box has to end up;
* two ``iris650_soarm100``, one per painted spawn zone.

Intended flow: both robots fly to the pick table, each closes on the handle
on its own side, together they lift the box and fly it to the drop table, and
set it down inside the green rectangle.

Identical robots on purpose, unlike ``AerialHandover``: a two-point lift only
makes sense when both ends behave the same.

Not implemented yet: grasping, the shared-load dynamics, and any reward that
knows about them. A true rigid two-point carry needs a closed kinematic loop
(two fixed joints onto one body), which is its own piece of engineering.
Reward and success are still ``CollaborativeAerialEnv``'s placeholder -- hold
position near a per-agent goal -- with the goals set above each robot's
handle *at the drop table*, so the placeholder at least points the right way.
"""

from typing import List, Sequence

import numpy as np
from omni.isaac.core.objects import FixedCuboid, VisualCuboid

from aetos_multiam.objects import CarryBox
from aetos_multiam.scenes import WarehouseArena

from .collaborative_base import CollaborativeAerialEnv

SPAWN_HEIGHT_M = 1.2

#: Both tables stand at x = 0, between the painted zones, stacked along y.
PICK_TABLE_Y_M = 0.0
DROP_TABLE_Y_M = 2.50
TABLE_TOP_Z_M = 0.45
TABLE_SIZE_M = (1.40, 0.80, 0.04)
TABLE_LEG_M = 0.06

#: Painted target on the drop table: the box's footprint plus a margin.
TARGET_MARGIN_M = 0.08
TARGET_LINE_HALF_WIDTH_M = 0.012
GREEN = (0.10, 0.75, 0.25)

#: How high above a handle a drone has to sit to reach down to it.
HANDLE_HOVER_M = 0.75

STEEL = (0.55, 0.58, 0.60)
DARK = (0.18, 0.20, 0.22)


class AerialTransport(CollaborativeAerialEnv):
    # -- scene ---------------------------------------------------------------

    def _build_arena(self) -> WarehouseArena:
        return WarehouseArena(floor_color=WarehouseArena.CONCRETE, gantry=False)

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        centers = self.arena.spawn_zone_centers
        cx, cy = centers[index % len(centers)]
        return (cx, cy, SPAWN_HEIGHT_M)

    # -- waypoints the flow needs -------------------------------------------

    @property
    def pick_table_center(self) -> Sequence[float]:
        return (0.0, PICK_TABLE_Y_M, TABLE_TOP_Z_M)

    @property
    def drop_table_center(self) -> Sequence[float]:
        return (0.0, DROP_TABLE_Y_M, TABLE_TOP_Z_M)

    def handle_position(self, index: int, on_drop_table: bool = False) -> Sequence[float]:
        """Where handle ``index`` is (or has to end up), in world coordinates.

        Index 0 is the -x handle, which belongs to the robot spawned in the
        -x zone -- the two orderings are deliberately the same.
        """
        dx, _, dz = self.carry_box.handle_local_offsets()[index]
        y = DROP_TABLE_Y_M if on_drop_table else PICK_TABLE_Y_M
        return (dx, y, TABLE_TOP_Z_M + dz)

    # -- prims ---------------------------------------------------------------

    def _load_objects(self) -> List[str]:
        ns = self.template_env_ns
        self._table_prims("pick", PICK_TABLE_Y_M)
        self._table_prims("drop", DROP_TABLE_Y_M)
        self._target_rectangle()

        self.carry_box = CarryBox(name="carry_box")
        self.carry_box.translation = self.carry_box.resting_translation(
            0.0, PICK_TABLE_Y_M, TABLE_TOP_Z_M + 0.002)
        self.carry_box.spawn(ns)
        if self.cfg.task.get("filter_box_arm_contacts", False):
            self._filter_box_arm_contacts()

        # Everything here is inside the template env and gets cloned.
        return []

    #: SO101 links the box may not collide with when contacts are filtered.
    ARM_LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")

    def _filter_box_arm_contacts(self) -> None:
        """Box <-> arm-link collisions off (authored before the sim starts).

        For a KINEMATICALLY carried box: with the arm dynamics frozen, every arm
        link weighs ~1 g, and contact against a teleported 1 kg box bends the
        elbow ~1 rad and drags the drones off their setpoints. Grasp contact is
        then judged geometrically instead (see
        scripts/collect_aerial_transport_demo.py)."""
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdPhysics

        ns = self.template_env_ns
        rel = UsdPhysics.FilteredPairsAPI.Apply(
            get_current_stage().GetPrimAtPath(f"{ns}/{self.carry_box.name}")).CreateFilteredPairsRel()
        for i in range(self.num_agents):
            for link in self.ARM_LINKS:
                rel.AddTarget(f"{ns}/am_{i}_0/manipulator/{link}")

    def _table_prims(self, tag: str, cy: float) -> None:
        ns = self.template_env_ns
        w, d, t = TABLE_SIZE_M
        FixedCuboid(
            prim_path=f"{ns}/{tag}_table_top", name=f"{tag}_table_top",
            translation=(0.0, cy, TABLE_TOP_Z_M - t / 2.0),
            scale=np.array([w, d, t]), color=np.array(STEEL),
        )
        leg_h = TABLE_TOP_Z_M - t
        for sx in (-1, 1):
            for sy in (-1, 1):
                leg = f"{'n' if sx < 0 else 'p'}{'n' if sy < 0 else 'p'}"
                FixedCuboid(
                    prim_path=f"{ns}/{tag}_table_leg_{leg}", name=f"{tag}_table_leg_{leg}",
                    translation=(sx * (w / 2.0 - TABLE_LEG_M),
                                 cy + sy * (d / 2.0 - TABLE_LEG_M), leg_h / 2.0),
                    scale=np.array([TABLE_LEG_M, TABLE_LEG_M, leg_h]),
                    color=np.array(DARK),
                )

    def _target_rectangle(self) -> None:
        """Green outline on the drop table. Visual only -- a 2 mm lip under a
        box that is being set down is a contact bug waiting to happen, the
        same reason the arena's painted zones have no collider."""
        ns = self.template_env_ns
        hx = CarryBox.body_size_m[0] / 2.0 + TARGET_MARGIN_M
        hy = CarryBox.body_size_m[1] / 2.0 + TARGET_MARGIN_M
        w = TARGET_LINE_HALF_WIDTH_M
        z = TABLE_TOP_Z_M + 0.002
        for s in (-1, 1):
            tag = "n" if s < 0 else "p"
            VisualCuboid(
                prim_path=f"{ns}/drop_target_x{tag}", name=f"drop_target_x{tag}",
                translation=(s * hx, DROP_TABLE_Y_M, z),
                scale=np.array([2 * w, 2 * (hy + w), 0.002]), color=np.array(GREEN),
            )
            VisualCuboid(
                prim_path=f"{ns}/drop_target_y{tag}", name=f"drop_target_y{tag}",
                translation=(0.0, DROP_TABLE_Y_M + s * hy, z),
                scale=np.array([2 * hx, 2 * w, 0.002]), color=np.array(GREEN),
            )

    def _setup_object_references(self) -> None:
        self.carry_box.initialize(f"{self.env_ns}/env_.*/carry_box")

    # -- task ----------------------------------------------------------------

    def _goal_position(self, index: int) -> Sequence[float]:
        hx, hy, hz = self.handle_position(index, on_drop_table=True)
        return (hx, hy, hz + HANDLE_HOVER_M)
