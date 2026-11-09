"""Warehouse arena -- an enclosed industrial bay with two marked spawn zones
and a gantry between them.

Layout follows the MuJoCo scene this was specified from
(``industrial_warehouse_mujoco/build_scene.py``), with one deliberate change:
there the gantry sits *behind* both yellow zones and spans across them; here
it sits **between** them, because the two zones are the two aerial
manipulators' start areas and the gantry is what they meet at. So the zones
are separated along X and the gantry's beam runs along Y through the origin,
putting its four posts outside both zones and nothing in the lanes.

Everything is a static box. The MuJoCo source's background dressing (pallet
racking, cartons, the rear roll-up door) is left out -- it is scenery, not
something a task touches. The floor is ``Arena``'s ground plane, i.e. Isaac's
grid plane tinted the way AM-Bench tints it; the concrete texture the MuJoCo
scene bakes has no equivalent here without authoring a material.

Sizes below are **half**-extents, matching the MJCF they came from;
``FixedCuboid``/``VisualCuboid`` take full extents, so ``_box`` doubles them.
"""

from typing import List, Sequence, Tuple

import numpy as np
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects import FixedCuboid, VisualCuboid

from .arena_base import Arena

# MJCF material colours, as RGB.
WALL = (0.58, 0.61, 0.62)
TRIM = (0.25, 0.29, 0.31)
BLUE = (0.12, 0.25, 0.32)
STEEL = (0.53, 0.59, 0.62)
YELLOW = (1.0, 0.74, 0.025)
DARK = (0.10, 0.12, 0.13)


def _s(sign: int) -> str:
    """USD prim names reject '-' and '+', so signs become letters."""
    return "n" if sign < 0 else "p"


class WarehouseArena(Arena):
    """Enclosed bay, two yellow spawn zones, one gantry between them."""

    #: Interior half-extents (x, y) and wall height, metres. 18 x 20 x 5.4 m.
    interior_half = (9.0, 10.0)
    wall_height = 5.4
    wall_half_thickness = 0.12

    #: Centres of the two spawn zones; a task reads these instead of hardcoding.
    spawn_zone_centers: Tuple[Tuple[float, float], ...] = ((-4.5, 0.0), (4.5, 0.0))
    #: Half-extents of one zone's painted outline (3.8 m x 4.0 m overall).
    spawn_zone_half = (1.9, 2.0)
    zone_line_half_width = 0.045

    #: ``None`` keeps AM-Bench's own grid plane as the floor, which is what
    #: this scene was asked for. An RGB instead (concrete is
    #: ``(0.70, 0.70, 0.68)``) *replaces* it with a plain slab covering the
    #: bay -- replaces, not covers: a slab laid on top of the grid plane
    #: z-fights with it right across the floor at room distances, and hiding
    #: the grid plane's visuals is not reliable (the grid is not drawn by a
    #: Mesh prim). The slab is then the collider, so the bay's floor is finite
    #: -- fine for an enclosed room, and nothing goes outside it.
    floor_color = None
    floor_thickness = 0.2

    #: Gantry: posts at (+-post_half_span_x, +-post_half_span_y), beam top at ~4.6 m.
    gantry_post_x = 0.8
    gantry_post_y = 4.0
    gantry_beam_z = 4.43
    #: Where task objects will hang from. No prims -- positions only, for later.
    hang_anchor_positions: Tuple[Tuple[float, float, float], ...] = (
        (0.0, -1.6, 4.25),
        (0.0, 1.6, 4.25),
    )

    #: Plain concrete, the usual value for ``floor_color``.
    CONCRETE = (0.70, 0.70, 0.68)

    #: The gantry is scenery for whatever hangs off it. Tasks that put their
    #: own fixture between the zones turn it off.
    gantry = True

    def __init__(self, prim_path: str = "warehouse", floor_color=..., gantry: bool = None):
        self.prim_path = prim_path  # relative to the template env namespace
        if floor_color is not ...:
            self.floor_color = floor_color
        if gantry is not None:
            self.gantry = bool(gantry)
        self._root = None
        self._n = 0

    # ------------------------------------------------------------------ util
    def _box(self, name: str, center: Sequence[float], half: Sequence[float],
             color: Sequence[float], collide: bool = True):
        cls = FixedCuboid if collide else VisualCuboid
        self._n += 1
        cls(
            prim_path=f"{self._root}/{name}",
            name=f"{self.prim_path}_{name}_{self._n}",
            translation=tuple(float(v) for v in center),
            scale=np.array([2.0 * float(v) for v in half]),
            color=np.array(color),
        )

    # ----------------------------------------------------------------- spawn
    def spawn(self, template_env_ns: str) -> List[str]:
        self._root = f"{template_env_ns}/{self.prim_path}"
        self._n = 0

        global_paths = self._spawn_floor(template_env_ns)
        self._spawn_shell()
        self._spawn_zones()
        self._spawn_gantry()
        return global_paths

    def _spawn_floor(self, template_env_ns: str) -> List[str]:
        if self.floor_color is None:
            return super().spawn(template_env_ns)
        hx, hy = self.interior_half
        half_t = self.floor_thickness / 2.0
        self._box("floor", (0.0, 0.0, -half_t), (hx, hy, half_t), self.floor_color)
        # create_ground_plane would have brought this; without it the bay is
        # lit by whatever the caller adds and nothing else.
        prim_utils.create_prim(f"{self._root}/SunLight", "DistantLight",
                               attributes={"inputs:intensity": 600.0})
        # No global paths: unlike the shared ground plane, this floor lives
        # inside the template env and gets cloned, so each env has its own.
        return []

    def _spawn_shell(self) -> None:
        hx, hy = self.interior_half
        t = self.wall_half_thickness
        hz = self.wall_height / 2.0

        for sx in (-1, 1):
            x = sx * hx
            self._box(f"wall_x{_s(sx)}", (x, 0.0, hz), (t, hy, hz), WALL)
            self._box(f"base_x{_s(sx)}", (x, 0.0, 0.28), (t + 0.02, hy, 0.28), TRIM)
            # Pilasters, inset just off the wall face so they read as columns.
            for i, y in enumerate((-8.0, -4.0, 0.0, 4.0, 8.0)):
                self._box(f"pilaster_x{_s(sx)}_{i}", (x * 0.987, y, hz), (t, 0.13, hz), BLUE)

        for sy in (-1, 1):
            y = sy * hy
            self._box(f"wall_y{_s(sy)}", (0.0, y, hz), (hx, t, hz), WALL)
            self._box(f"base_y{_s(sy)}", (0.0, y, 0.28), (hx, t + 0.02, 0.28), TRIM)
            for i, z in enumerate((1.8, 3.6, 5.1)):
                self._box(f"seam_y{_s(sy)}_{i}", (0.0, y * 0.987, z), (hx, 0.018, 0.018), TRIM)

    def _spawn_zones(self) -> None:
        """Painted outlines. Visual only -- the ground stays the one collision
        surface, exactly as the MJCF does it (``contype=0 conaffinity=0``);
        a 2 mm lip under a landing gear is a contact bug waiting to happen."""
        hx, hy = self.spawn_zone_half
        w = self.zone_line_half_width
        z = 0.002
        for idx, (cx, cy) in enumerate(self.spawn_zone_centers):
            for s in (-1, 1):
                self._box(f"zone{idx}_x{_s(s)}", (cx + s * hx, cy, z),
                          (w, hy + w, 0.001), YELLOW, collide=False)
                self._box(f"zone{idx}_y{_s(s)}", (cx, cy + s * hy, z),
                          (hx, w, 0.001), YELLOW, collide=False)

    def _spawn_gantry(self) -> None:
        if not self.gantry:
            return
        px, py, bz = self.gantry_post_x, self.gantry_post_y, self.gantry_beam_z
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = sx * px, sy * py
                tag = f"{_s(sx)}{_s(sy)}"
                self._box(f"gantry_foot_{tag}", (x, y, 0.06), (0.34, 0.40, 0.06), DARK)
                self._box(f"gantry_post_{tag}", (x, y, 2.25), (0.12, 0.12, 2.25), BLUE)
                # Impact guard at the base -- the one yellow thing not on the floor.
                self._box(f"gantry_guard_{tag}", (x, y, 0.35), (0.135, 0.135, 0.22), YELLOW)
            # Main beam, spanning the gap between the two zones.
            self._box(f"gantry_beam_x{_s(sx)}", (sx * px, 0.0, bz), (0.13, py + 0.13, 0.17), BLUE)
            self._box(f"gantry_rail_x{_s(sx)}", (sx * px, 0.0, bz + 0.21), (0.055, py - 0.1, 0.045), STEEL)
        for sy in (-1, 1):
            self._box(f"gantry_side_y{_s(sy)}", (0.0, sy * py, bz), (px + 0.15, 0.13, 0.17), BLUE)
        for idx, (ax, ay, az) in enumerate(self.hang_anchor_positions):
            self._box(f"trolley_{idx}", (ax, ay, az + 0.12), (0.86, 0.23, 0.10), YELLOW)
