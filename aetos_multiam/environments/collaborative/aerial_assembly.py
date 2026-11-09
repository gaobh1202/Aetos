"""AerialAssembly -- two AMs each fetch a gear and insert it into a shared base.

Scene (``WarehouseArena``, concrete floor, no gantry):

* a **pedestal** standing between the two painted spawn zones, the same
  footprint as the plate it carries, with IsaacLab's ``factory_gear_base``
  bolted on top -- three shafts, pointing up;
* a lower **table** beside it, one gear resting on each of its two ends;
* two ``iris650_soarm100``, one per zone.

Each agent flies to its own end of the table, picks up the gear there, carries
it to the pillar and drops it down its shaft. The two gears are the *medium*
and *small* ones, which sit on neighbouring shafts 30.5 mm apart and mesh with
each other once both are seated -- so "both inserted" and "meshed" are the same
end state, not two separate conditions.

Why medium+small rather than large+medium: every gear's bore is 10 mm, so
relative insertion clearance is identical whichever pair is used, and the
smaller pair leaves more of the base plate free.

``task.gear_scale`` defaults to **4**, not 1: at IsaacLab's own sizes the
gears are 42 mm and 22 mm across, which is invisible next to a 650 mm drone
in an 18 m bay. Scaling them up does not make them harder to pick up, because
what the gripper closes on is not the gear -- it is a fixed-size grab post
standing on each gear's axis (``objects/factory_gear.py``), 30 mm across
whatever the gear does, which the SO-ARM jaw reaches at about +6.5 deg of its
110 deg travel (``debug/measure_gripper_range.py``).

No grasp controller and no insertion reward yet -- this still inherits
``CollaborativeAerialEnv``'s placeholder (hold position near a per-agent
goal), with the goals moved to sit above each agent's own shaft.

The gear colliders are SDF, which needs the GPU PhysX pipeline;
``cfg/base/sim_base.yaml`` already asks for it. See
``objects/factory_gear.py`` for what happens if it doesn't.
"""

from typing import List, Sequence

import numpy as np
from omni.isaac.core.objects import FixedCuboid

from aetos_multiam.objects import FactoryGear, FactoryGearBase
from aetos_multiam.objects.factory_gear import PLATE_SIZE_XY
from aetos_multiam.scenes import WarehouseArena

from .collaborative_base import CollaborativeAerialEnv

SPAWN_HEIGHT_M = 1.2

# Heights are set so the grab posts land where a drone hovering at
# SPAWN_HEIGHT_M can reach them: its arm hangs roughly 0.3 m below the body.
PILLAR_HEIGHT_M = 0.65     # its footprint matches the base plate exactly, see _load_objects

TABLE_Y_M = -1.10          # beside the pillar, on the near side of the bay
TABLE_TOP_Z_M = 0.45       # "lower than the pillar"
TABLE_SIZE_M = (1.60, 0.60, 0.04)
TABLE_LEG_M = 0.06
GEAR_EDGE_X_M = 0.55       # how far out on the table each gear sits

#: Which agent fetches which gear. Index is the agent, so agent 0 (the -x
#: zone) gets the gear parked at -x on the table.
AGENT_GEARS = ("medium", "small")

STEEL = (0.55, 0.58, 0.60)
DARK = (0.18, 0.20, 0.22)


class AerialAssembly(CollaborativeAerialEnv):
    def __init__(self, cfg, headless):
        self.gear_scale = float(cfg.task.get("gear_scale", 4.0))
        super().__init__(cfg, headless)

    # -- scene ---------------------------------------------------------------

    def _build_arena(self) -> WarehouseArena:
        return WarehouseArena(floor_color=WarehouseArena.CONCRETE, gantry=False)

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        """One agent per painted zone. Read off the arena rather than repeated
        here, so moving a zone moves the robot that starts in it."""
        centers = self.arena.spawn_zone_centers
        cx, cy = centers[index % len(centers)]
        return (cx, cy, SPAWN_HEIGHT_M)

    def _gear_pickup_xy(self, index: int) -> Sequence[float]:
        side = -1.0 if index == 0 else 1.0
        return (side * GEAR_EDGE_X_M, TABLE_Y_M)

    def _load_objects(self) -> List[str]:
        ns = self.template_env_ns
        s = self.gear_scale

        # Pedestal with the same footprint as the plate it carries, so the two
        # read as one piece of furniture rather than a plate balanced on a post.
        FixedCuboid(
            prim_path=f"{ns}/gear_pillar", name="gear_pillar",
            translation=(0.0, 0.0, PILLAR_HEIGHT_M / 2.0),
            scale=np.array([PLATE_SIZE_XY[0] * s, PLATE_SIZE_XY[1] * s, PILLAR_HEIGHT_M]),
            color=np.array(STEEL),
        )
        self._table_prims()

        self.gear_base = FactoryGearBase(
            name="gear_base", translation=(0.0, 0.0, PILLAR_HEIGHT_M), scale=s)
        self.gear_base.spawn(ns)

        self.gears: List[FactoryGear] = []
        for i, size in enumerate(AGENT_GEARS):
            gear = FactoryGear(size, name=f"gear_{i}_{size}", scale=s)
            px, py = self._gear_pickup_xy(i)
            gear.translation = gear.resting_translation(px, py, TABLE_TOP_Z_M)
            gear.spawn(ns)
            self.gears.append(gear)

        # Nothing here is shared between envs -- it all gets cloned.
        return []

    def _table_prims(self) -> None:
        ns = self.template_env_ns
        w, d, t = TABLE_SIZE_M
        FixedCuboid(
            prim_path=f"{ns}/gear_table_top", name="gear_table_top",
            translation=(0.0, TABLE_Y_M, TABLE_TOP_Z_M - t / 2.0),
            scale=np.array([w, d, t]), color=np.array(STEEL),
        )
        leg_h = TABLE_TOP_Z_M - t
        for sx in (-1, 1):
            for sy in (-1, 1):
                tag = f"{'n' if sx < 0 else 'p'}{'n' if sy < 0 else 'p'}"
                FixedCuboid(
                    prim_path=f"{ns}/gear_table_leg_{tag}", name=f"gear_table_leg_{tag}",
                    translation=(sx * (w / 2.0 - TABLE_LEG_M),
                                 TABLE_Y_M + sy * (d / 2.0 - TABLE_LEG_M),
                                 leg_h / 2.0),
                    scale=np.array([TABLE_LEG_M, TABLE_LEG_M, leg_h]),
                    color=np.array(DARK),
                )

    def _setup_object_references(self) -> None:
        self.gear_base.initialize(f"{self.env_ns}/env_.*/gear_base")
        for i, gear in enumerate(self.gears):
            gear.initialize(f"{self.env_ns}/env_.*/{gear.name}")

    # -- task ----------------------------------------------------------------

    def _goal_position(self, index: int) -> Sequence[float]:
        """Hover point above this agent's own shaft -- where it has to be to
        lower its gear onto the base."""
        sx, sy = self.gear_base.shaft_world_xy(AGENT_GEARS[index])
        return (sx, sy, PILLAR_HEIGHT_M + 0.55)
