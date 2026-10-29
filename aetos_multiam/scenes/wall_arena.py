"""Wall arena for the ButtonGrasp task (robosuite ``TableArena`` analog).

AM-bench authors its wall as a full USD asset
(``ambench.scenes.wall.spawn_wall``, IsaacLab ``RigidObjectCfg``); since this
benchmark doesn't depend on IsaacLab, the wall here is just a static cuboid
via ``omni.isaac.core``'s ``FixedCuboid`` -- visually simpler, but all that's
needed to mount a ``Button`` object against.
"""

from typing import List, Sequence

from omni.isaac.core.objects import FixedCuboid

from .arena_base import Arena


class WallArena(Arena):
    def __init__(
        self,
        wall_translation: Sequence[float] = (2.0, 0.0, 1.0),
        wall_size: Sequence[float] = (0.1, 2.0, 2.0),
    ):
        self.wall_translation = wall_translation
        self.wall_size = wall_size
        self.wall_prim_path = "wall"  # relative to the template env namespace

    def spawn(self, template_env_ns: str) -> List[str]:
        global_paths = super().spawn(template_env_ns)
        wall_prim_path = f"{template_env_ns}/{self.wall_prim_path}"
        FixedCuboid(
            prim_path=wall_prim_path,
            translation=self.wall_translation,
            scale=self.wall_size,
        )
        return global_paths
