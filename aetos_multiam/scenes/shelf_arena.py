"""Shelf arena -- migrated from raids-am-simulator's shelf_grasp scene
(``raids_am_Scene/build_isaacsim_scene.py``). The shelf is static furniture
(flipped kinematic, same as that script does), analogous to ``WallArena``'s
wall. The hand-tuned grid-texture ground plane from the source script is
skipped here in favor of ``Arena``'s plain default ground -- cosmetic only,
not needed for physics or task logic.
"""

from pathlib import Path
from typing import List

import omni.isaac.core.utils.prims as prim_utils

from aetos_multiam.utils import make_kinematic, normalize_rigid_root_xform

from .arena_base import Arena

_ASSETS = Path(__file__).resolve().parent.parent / "objects" / "assets"
SHELF_USD = str(_ASSETS / "shelf_chrome" / "model_B00NUS53CY_69323.usd")


class ShelfArena(Arena):
    """4-tier chrome shelf ("4 Tier Assembled Shelf, Chrome", ~0.91m x
    0.35m x 1.36m, widened 1.5x along local X), same hand-tuned pose/scale
    as the source scene."""

    shelf_prim_path = "shelf"  # relative to the template env namespace
    shelf_translation = (0.0, 0.0, 0.0)
    shelf_scale = (1.5, 1.0, 1.0)  # local-X widening, hand-set in the source scene
    shelf_root_body = "E_RootNode001_2"

    def spawn(self, template_env_ns: str) -> List[str]:
        global_paths = super().spawn(template_env_ns)
        prim_path = f"{template_env_ns}/{self.shelf_prim_path}"
        prim_utils.create_prim(prim_path, usd_path=SHELF_USD, translation=self.shelf_translation)
        normalize_rigid_root_xform(f"{prim_path}/{self.shelf_root_body}", scale=self.shelf_scale)
        n_kinematic = make_kinematic(prim_path)
        if n_kinematic == 0:
            raise RuntimeError(f"expected at least one RigidBodyAPI prim under {prim_path}")
        return global_paths
