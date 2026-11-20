"""Two-door cabinet -- the fixture in ``AerialCabinetGrasp``.

Asset: PalatialSim/palatial-nv-asset-library asset ``D00241`` (CC-BY-4.0),
vendored as-is under ``objects/assets/cabinet/`` -- the same library and the
same folder layout as ``white_bucket`` (``D00436``). One articulation,
three bodies: the carcass (``part_2``) and two doors hinged on it about
vertical axes (``part_1`` at -x, ``part_0`` at +x). The open front faces the
asset's own **-y**. Inside are three shelf boards, so three compartments.

Everything below marked "asset frame" was measured off the USD's own meshes,
unscaled, in the frame of its default prim. Four things about the file are
not what one would guess, and ``spawn`` deals with each:

* **Joint angle 0 is the doors wide open**, not shut -- the door meshes are
  authored standing out along -y. Shut is each joint's *lower* limit (-93 and
  -88 deg). ``spawn`` authors that as the joints' initial state and
  ``reset_doors`` re-applies it, so a scene never starts with the doors open
  by accident.
* **The carcass is a free 25 kg body.** A drone pulling on a door would drag
  the whole cabinet across the floor, so it is welded to the world.
* **The +x door's collider is one convex hull around panel and handle
  together**, which fills in the very gap a gripper has to close into (the
  -x door is decomposed into 15 pieces and is fine). That hull is switched
  off and replaced with two boxes, panel and handle bar.
* **Door inertia is authored as numbers**, which a scale on the prim does not
  touch. It is cleared so PhysX derives it from the (scaled) colliders.

Door index convention, used by every method here: **0 is the -x door, 1 is
the +x door** -- the same order ``WarehouseArena`` lists its spawn zones in,
so agent ``i`` standing in zone ``i`` faces door ``i``.
"""

import math
from pathlib import Path
from typing import Sequence, Tuple

import isaacsim.core.utils.prims as prim_utils
import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from aetos_multiam.utils import pin_body_to_world

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "cabinet" / "simready_usd" / "D00241.usd")

#: The articulation root's prim name inside the asset.
ARTICULATION_NAME = "textured_mesh_6a741da85d45b936d647e876"
#: That prim's own z offset under the default prim. Collider prims are
#: authored in *its* frame, the measurements below in the default prim's.
_ARTICULATION_Z_M = 0.005697518587112427


class Cabinet(AetosObject):
    """A cabinet whose two doors have to be pulled open by their handles."""

    usd_path = USD_PATH
    has_joints = True

    body_name = "part_2"
    #: Indexed by door: 0 = -x, 1 = +x.
    door_body_names = ("part_1", "part_0")
    door_joint_names = ("joint_part_2__part_1", "joint_part_2__part_0")
    #: Joint angle with the door shut, degrees. Open is 0 for both.
    door_closed_deg = (-93.0, -88.0)
    #: Which way the door body turns about +z per unit of joint angle. The -x
    #: joint's frames are authored flipped 180 deg about y, so it is -1.
    door_yaw_sign = (-1.0, 1.0)

    # -- asset frame, unscaled, metres ---------------------------------------
    #: Hinge axis positions (x, y).
    hinge_xy = ((-0.41474, -0.15193), (0.41686, -0.14800))
    #: Where each door body's own origin is authored. It is the hinge's anchor
    #: on the door, and it does **not** coincide with ``hinge_xy``, the anchor
    #: on the carcass -- the file is off by 5 mm in x and about 30 mm in y. The
    #: solver closes that gap the moment the simulation starts, so a simulated
    #: door sits that much further out than its mesh is drawn in the USD.
    door_origin_xy = ((-0.40974, -0.12093), (0.41186, -0.12124))
    #: Centre of each handle bar **with the door open** (joint angle 0), as
    #: drawn in the USD -- ``handle_position`` applies the correction above. The
    #: bar is vertical, 23 x 23 mm in section and 195 mm long, standing 37 mm
    #: proud of the door's outer face, near its free edge.
    handle_center_open = ((-0.4265, -0.5120, 0.6895), (0.4085, -0.5120, 0.6895))
    handle_bar_section_m = 0.023
    handle_bar_length_m = 0.195
    #: Clear interior: x and y extents, and each compartment's (floor, ceiling).
    interior_x = (-0.349, 0.356)
    interior_y = (-0.113, 0.276)        # front opening edge, back panel
    compartment_z = ((0.106, 0.468), (0.508, 0.880), (0.921, 1.253))
    #: Outer size (x, y with doors shut, z).
    outer_size_m = (0.94, 0.52, 1.32)

    # +x door, for the replacement colliders: panel slab and handle bar boxes
    # as (min, max) corners with the door open.
    _door1_panel_box = ((0.362, -0.537, 0.087), (0.390, -0.111, 1.268))
    _door1_handle_box = ((0.390, -0.5236, 0.592), (0.427, -0.5004, 0.787))

    #: Angular damping on the hinges, N*m*s/deg -- twice the asset's own 0.1.
    #: A door let go of mid-swing coasts to a stop within a few degrees
    #: instead of carrying on to its limit, while 8 N on the handle still
    #: opens it in about 3 s (``scripts/check_cabinet_grasp.py``).
    hinge_damping = 0.2

    def __init__(self, name: str = "cabinet", translation=(0.0, 0.0, 0.0),
                 yaw_deg: float = 0.0, scale: float = 1.0):
        super().__init__(name, translation=translation, yaw_deg=yaw_deg, scale=scale)
        self.yaw_deg = float(yaw_deg)

    # -- geometry the task needs --------------------------------------------

    def to_parent(self, local_xyz: Sequence[float]) -> Tuple[float, float, float]:
        """Asset-frame point (unscaled) -> the frame ``translation`` is given
        in, i.e. the env's own, after this object's scale, yaw and offset."""
        x, y, z = (self.scale * float(v) for v in local_xyz)
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        tx, ty, tz = self.translation
        return (tx + c * x - s * y, ty + s * x + c * y, tz + z)

    def _direction_to_parent(self, dx: float, dy: float) -> Tuple[float, float]:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return (c * dx - s * dy, s * dx + c * dy)

    def _door_yaw_rad(self, door: int, open_fraction: float) -> float:
        """How far the door body is turned from its authored (open) pose."""
        joint_deg = self.door_closed_deg[door] * (1.0 - float(open_fraction))
        return math.radians(self.door_yaw_sign[door] * joint_deg)

    def handle_position(self, door: int, open_fraction: float = 0.0) -> Tuple[float, float, float]:
        """Centre of a door's handle bar, in the env frame.

        ``open_fraction`` 0 is shut, 1 is fully open; anything between is the
        point on the arc the handle sweeps, which is what a pull trajectory
        has to follow.
        """
        hx, hy = self.hinge_xy[door]
        px, py, pz = self.handle_center_open[door]
        a = self._door_yaw_rad(door, open_fraction)
        # Offset from the door body's own origin, which the hinge pins to
        # hinge_xy -- see door_origin_xy for why those are two different points.
        ox, oy = self.door_origin_xy[door]
        rx, ry = px - ox, py - oy
        return self.to_parent((
            hx + math.cos(a) * rx - math.sin(a) * ry,
            hy + math.sin(a) * rx + math.cos(a) * ry,
            pz,
        ))

    def handle_normal(self, door: int, open_fraction: float = 0.0) -> Tuple[float, float]:
        """Unit (x, y) pointing out of the door's outer face at the handle --
        the direction a gripper has to come in *against*. Straight out of the
        cabinet front with the door shut, sideways once it is open."""
        outward_open = -1.0 if door == 0 else 1.0     # asset +-x with the door open
        a = self._door_yaw_rad(door, open_fraction)
        return self._direction_to_parent(math.cos(a) * outward_open, math.sin(a) * outward_open)

    def handle_approach_position(self, door: int, standoff: float,
                                 open_fraction: float = 0.0) -> Tuple[float, float, float]:
        """A point ``standoff`` metres out from the handle along its normal."""
        hx, hy, hz = self.handle_position(door, open_fraction)
        nx, ny = self.handle_normal(door, open_fraction)
        return (hx + standoff * nx, hy + standoff * ny, hz)

    @property
    def front_normal(self) -> Tuple[float, float]:
        """Unit (x, y) pointing out of the cabinet's open front."""
        return self._direction_to_parent(0.0, -1.0)

    def shelf_position(self, compartment: int, x_fraction: float = 0.5,
                       depth_m: float = 0.0) -> Tuple[float, float, float]:
        """A point on a compartment's floor, in the env frame.

        ``x_fraction`` runs 0..1 across the clear width from the -x side.
        ``depth_m`` is how far in from the front opening, in real (scaled)
        metres -- so a prop stays the same distance from the edge whatever
        scale the cabinet is spawned at.
        """
        x0, x1 = self.interior_x
        floor_z = self.compartment_z[compartment][0]
        return self.to_parent((
            x0 + float(x_fraction) * (x1 - x0),
            self.interior_y[0] + depth_m / self.scale,
            floor_z,
        ))

    def compartment_clear_size(self, compartment: int) -> Tuple[float, float, float]:
        """Clear (width, depth, height) of a compartment at this scale."""
        z0, z1 = self.compartment_z[compartment]
        return (
            self.scale * (self.interior_x[1] - self.interior_x[0]),
            self.scale * (self.interior_y[1] - self.interior_y[0]),
            self.scale * (z1 - z0),
        )

    # -- spawn ---------------------------------------------------------------

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        root = f"{prim_path}/{ARTICULATION_NAME}"
        stage = get_current_stage()

        pin_body_to_world(f"{root}/{self.body_name}")

        for joint_name, closed_deg in zip(self.door_joint_names, self.door_closed_deg):
            joint = stage.GetPrimAtPath(f"{root}/Joints/{joint_name}")
            if not joint.IsValid():
                raise RuntimeError(f"Cabinet: door joint not found: {joint.GetPath()}")
            state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
            state.CreatePositionAttr(float(closed_deg))
            state.CreateVelocityAttr(0.0)
            UsdPhysics.DriveAPI(joint, "angular").GetDampingAttr().Set(float(self.hinge_damping))

        for body_name in self.door_body_names:
            # The values MassAPI documents as "not authored, derive it" -- a
            # blocked attribute is not the same thing to PhysX, which then
            # rejects the body's inertia outright.
            mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{root}/{body_name}"))
            mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.0))
            mass.GetPrincipalAxesAttr().Set(Gf.Quatf(0.0, 0.0, 0.0, 0.0))
            mass.GetCenterOfMassAttr().Set(Gf.Vec3f(float("-inf")))

        self._rebuild_plus_x_door_colliders(root)
        return prim_path

    def _rebuild_plus_x_door_colliders(self, root: str) -> None:
        stage = get_current_stage()
        colliders = f"{root}/{self.door_body_names[1]}/Colliders_00"
        hull = stage.GetPrimAtPath(f"{colliders}/Collider_0")
        if not hull.IsValid():
            raise RuntimeError(f"Cabinet: expected collider not found: {colliders}/Collider_0")
        UsdPhysics.CollisionAPI(hull).CreateCollisionEnabledAttr(False)

        for tag, (lo, hi) in (("panel", self._door1_panel_box), ("handle", self._door1_handle_box)):
            size = [h - l for l, h in zip(lo, hi)]
            center = [(l + h) / 2.0 for l, h in zip(lo, hi)]
            center[2] -= _ARTICULATION_Z_M
            box = prim_utils.create_prim(
                f"{colliders}/aetos_{tag}_box", "Cube",
                translation=tuple(center), scale=tuple(size), attributes={"size": 1.0},
            )
            UsdPhysics.CollisionAPI.Apply(box)
            # Collision only: the door's own mesh is what gets drawn.
            UsdGeom.Imageable(box).CreatePurposeAttr(UsdGeom.Tokens.guide)

    # -- runtime -------------------------------------------------------------

    def initialize(self, prim_paths_expr: str) -> None:
        super().initialize(f"{prim_paths_expr}/{ARTICULATION_NAME}")
        dof_names = list(self._view._dof_names)
        self.door_joint_indices = torch.tensor(
            [dof_names.index(name) for name in self.door_joint_names], device=self._view._device)
        self._closed_rad = torch.deg2rad(
            torch.tensor(self.door_closed_deg, device=self._view._device))

    def get_door_joint_positions(self) -> torch.Tensor:
        """Hinge angles in radians, ``(num_envs, 2)``, door 0 first."""
        q = self._view.get_joint_positions()
        return q.reshape(-1, q.shape[-1])[:, self.door_joint_indices]

    def get_door_open_fraction(self) -> torch.Tensor:
        """``(num_envs, 2)`` in 0..1 -- 0 shut, 1 against the open stop."""
        return (1.0 - self.get_door_joint_positions() / self._closed_rad).clamp(0.0, 1.0)

    def reset_doors(self, env_ids: torch.Tensor, open_fraction: float = 0.0) -> None:
        for door in range(len(self.door_joint_names)):
            self.set_door_open_fraction(env_ids, door, open_fraction)

    def set_door_open_fraction(self, env_ids: torch.Tensor, door: int, open_fraction) -> None:
        """Put one door at ``open_fraction`` (a float, or one value per env in
        ``env_ids``) and stop it there. A teleport, not a drive: this is how a
        reset shuts the doors, and how a scripted run turns a door it is
        holding by the handle."""
        fraction = torch.as_tensor(open_fraction, dtype=torch.float32,
                                   device=self._view._device).expand(len(env_ids))
        q = (self._closed_rad[door] * (1.0 - fraction.clamp(0.0, 1.0))).reshape(-1, 1)
        index = self.door_joint_indices[door:door + 1]
        self._view.set_joint_positions(q, env_indices=env_ids, joint_indices=index)
        self._view.set_joint_velocities(torch.zeros_like(q), env_indices=env_ids, joint_indices=index)
