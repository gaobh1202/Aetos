"""AerialHandover -- one AM fetches a hammer, passes it to the other, which bins it.

Scene (``WarehouseArena``, concrete floor, no gantry):

* two **tables** in the middle between the painted spawn zones, one behind
  the other along y -- the near one carries a hammer, the far one a tote box;
* ``iris650_soarm100`` in the -x zone, ``uaquad600_gripper`` in the +x zone.

Intended flow: the SO-ARM robot lifts the hammer off its table, the two meet
over the middle of the bay, the hammer changes hands, and the gripper robot
drops it into the tote. The three waypoints that flow needs --
``hammer_pickup_position``, ``handover_position``, ``box_drop_position`` --
are properties here rather than numbers buried in a reward, so whatever ends
up driving the robots reads them from one place.

Heterogeneous on purpose: a 5-DOF arm with a jaw handing off to a 4-DOF arm
with a parallel gripper is the case a same-robot team never exercises.

Not implemented yet: grasping, the handover itself, and any reward that
knows about them. Reward and success are still
``CollaborativeAerialEnv``'s placeholder (both agents hold position near a
per-agent goal), with both goals set to the handover point.
"""

from typing import List, Sequence

import numpy as np
from omni.isaac.core.objects import FixedCuboid

from aetos_multiam.objects import Hammer, ToteBox
from aetos_multiam.scenes import WarehouseArena

from .collaborative_base import CollaborativeAerialEnv

SPAWN_HEIGHT_M = 1.2

#: Both tables stand at x = 0, in the middle between the two painted zones
#: (which span |x| 2.6..6.4), stacked along y -- same arrangement the gear
#: pedestal and its table use. The exchange happens in the gap between them.
TABLE_Y_M = 1.10
TABLE_TOP_Z_M = 0.45
#: 1.2 along x, 0.6 along y: the tables present their *short* sides to each
#: other across that gap, so neither long edge blocks the lane between them.
TABLE_SIZE_M = (1.20, 0.60, 0.04)
TABLE_LEG_M = 0.06

HANDOVER_Z_M = 1.3
TOTE_SCALE = 1.0          # 600 x 400 mm, roomy for a 268 mm hammer

STEEL = (0.55, 0.58, 0.60)
DARK = (0.18, 0.20, 0.22)


class AerialHandover(CollaborativeAerialEnv):
    # -- scene ---------------------------------------------------------------

    def _build_arena(self) -> WarehouseArena:
        return WarehouseArena(floor_color=WarehouseArena.CONCRETE, gantry=False)

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        centers = self.arena.spawn_zone_centers
        cx, cy = centers[index % len(centers)]
        return (cx, cy, SPAWN_HEIGHT_M)

    @property
    def hammer_pickup_position(self) -> Sequence[float]:
        return (0.0, -TABLE_Y_M, TABLE_TOP_Z_M)

    @property
    def box_drop_position(self) -> Sequence[float]:
        return (0.0, TABLE_Y_M, TABLE_TOP_Z_M + TOTE_SCALE * ToteBox.SIZE_M[2])

    @property
    def handover_position(self) -> Sequence[float]:
        return (0.0, 0.0, HANDOVER_Z_M)

    def _load_objects(self) -> List[str]:
        ns = self.template_env_ns
        for sign, tag in ((-1, "pick"), (1, "drop")):
            self._table_prims(f"{tag}", sign * TABLE_Y_M)

        # Both props lie along x, the tables' long axis.
        self.hammer = Hammer(name="hammer", yaw_deg=0.0)
        hx, hy, _ = self.hammer_pickup_position
        self.hammer.translation = self.hammer.resting_translation(hx, hy, TABLE_TOP_Z_M)
        self.hammer.spawn(ns)

        self.tote = ToteBox(name="tote", scale=TOTE_SCALE, yaw_deg=0.0,
                            translation=(0.0, TABLE_Y_M, TABLE_TOP_Z_M + 0.002))
        self.tote.spawn(ns)

        # Everything here is inside the template env and gets cloned.
        return []

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

    def _setup_object_references(self) -> None:
        self.hammer.initialize(f"{self.env_ns}/env_.*/hammer")
        self.tote.initialize(f"{self.env_ns}/env_.*/tote")

    # -- task ----------------------------------------------------------------

    def _goal_position(self, index: int) -> Sequence[float]:
        """Both agents aim at the exchange point for now -- the pick and drop
        stages are not in the reward yet (see the module docstring)."""
        return self.handover_position
