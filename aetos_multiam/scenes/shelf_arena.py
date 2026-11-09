"""Shelf arena -- migrated from raids-am-simulator's shelf_grasp scene
(``raids_am_Scene/build_isaacsim_scene.py``). The shelf is static furniture
(flipped kinematic, same as that script does), analogous to ``WallArena``'s
wall. The hand-tuned grid-texture ground plane from the source script is
skipped here in favor of ``Arena``'s plain default ground -- cosmetic only,
not needed for physics or task logic.

``spawn_shelf``/``on_shelf`` are the same shelf as free functions, for tasks
that want it standing in a different arena (``AerialButtonGrasp`` and
``AerialBucketGrasp`` put it in the middle of ``WarehouseArena``).
"""

import math
from pathlib import Path
from typing import List, Sequence

import omni.isaac.core.utils.prims as prim_utils

from aetos_multiam.utils import make_kinematic, normalize_rigid_root_xform, yaw_deg_to_quat

from .arena_base import Arena

_ASSETS = Path(__file__).resolve().parent.parent / "objects" / "assets"
SHELF_USD = str(_ASSETS / "shelf_chrome" / "model_B00NUS53CY_69323.usd")

#: Hand-set in the source scene: widen the shelf 1.5x along its local X.
SHELF_SCALE = (1.5, 1.0, 1.0)
SHELF_ROOT_BODY = "E_RootNode001_2"
#: The shelf's open front faces local +y -- that is the side the source
#: scene parks its robot on, and the side every prop on it is reachable from.
SHELF_FRONT_LOCAL = (0.0, 1.0, 0.0)


def spawn_shelf(prim_path: str, translation=(0.0, 0.0, 0.0), yaw_deg: float = 0.0,
                scale: Sequence[float] = SHELF_SCALE) -> None:
    """Spawn the 4-tier chrome shelf ("4 Tier Assembled Shelf, Chrome",
    ~0.91 m x 0.35 m x 1.36 m before ``scale``) as static furniture."""
    prim_utils.create_prim(prim_path, usd_path=SHELF_USD, translation=translation,
                           orientation=yaw_deg_to_quat(yaw_deg))
    normalize_rigid_root_xform(f"{prim_path}/{SHELF_ROOT_BODY}", scale=scale)
    n_kinematic = make_kinematic(prim_path)
    if n_kinematic == 0:
        raise RuntimeError(f"expected at least one RigidBodyAPI prim under {prim_path}")


def on_shelf(local_xyz: Sequence[float], translation=(0.0, 0.0, 0.0),
             yaw_deg: float = 0.0) -> tuple:
    """Turn a prop offset measured against an unrotated shelf into a world
    position for a shelf spawned at ``translation`` with ``yaw_deg``.

    Lets the hand-tuned offsets read back from the source scene stay exactly
    as they were measured, however the shelf is placed afterwards.
    """
    x, y, z = local_xyz
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (translation[0] + x * c - y * s, translation[1] + x * s + y * c, translation[2] + z)


class ShelfArena(Arena):
    """Bare ground plus the shelf at the template env origin, unrotated --
    the source scene's own layout."""

    shelf_prim_path = "shelf"  # relative to the template env namespace
    shelf_translation = (0.0, 0.0, 0.0)
    shelf_scale = SHELF_SCALE
    shelf_root_body = SHELF_ROOT_BODY

    def spawn(self, template_env_ns: str) -> List[str]:
        global_paths = super().spawn(template_env_ns)
        spawn_shelf(f"{template_env_ns}/{self.shelf_prim_path}",
                    translation=self.shelf_translation, scale=self.shelf_scale)
        return global_paths
