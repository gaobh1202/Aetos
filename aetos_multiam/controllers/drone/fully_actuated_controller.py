"""Position controller for a platform that can push sideways without tilting.

``LeePositionController`` commands four DOFs: one thrust along body-z plus
three torques. Its attitude target is therefore *not* free -- it is derived
from the force it wants (``b3_des = -normalize(acc)``), which is why an
underactuated multirotor has to lean to translate.

A fixed-cant platform like AM-Bench's FA-Hexa does not have to. Its six
non-parallel rotors span a rank-6 wrench space, so force and torque can be
commanded independently. This controller does exactly that:

* position loop -> a full 3-vector force in world frame (the same PD law,
  just not projected onto one body axis),
* attitude loop -> whatever attitude was asked for, by default *level* at
  the commanded yaw, tracked independently of the force,
* the two concatenated into a 6-vector body wrench and hit with the
  pseudo-inverse of the 6xN allocation matrix.

Same structure as AM-Bench's ``PID6DOFController`` + ``ctrl_alloc(...,
fully_actuated=True)``, minus the integral terms -- the gains reused here are
the PD ones the yaml already carries for the Lee controller, so adding an
I-term would mean retuning all four gain vectors, not three lines.

**The lateral force is clamped** to ``max_lateral_force_n``. This is not
conservatism: the allocation is exact only while every rotor stays inside
``[0, max_thrust]``, and past that the solver hands PhysX a wrench it then
silently clips, which couples back into attitude. FA-Hexa can hold 15.7 N
(0.24 g) sideways at hover from its worst azimuth, so a 1 m position error
at ``position_gain=4.0`` asks for roughly twice what the airframe has.
Clamping turns that into a slower straight translation instead of a tilt.
"""

import torch
import torch.nn as nn

from omni_drones.controllers import LeePositionController
from omni_drones.utils.torch import quat_rotate_inverse, quaternion_to_rotation_matrix

from .allocation import build_allocation_matrix

__all__ = ["FullyActuatedPositionController", "make_position_controller"]


class FullyActuatedPositionController(LeePositionController):
    """Drop-in for ``LeePositionController``: same ``compute(...)`` signature.

    Takes the drone itself rather than ``drone.params``, because the rotor
    geometry the allocation needs (positions, canted axes, drag ratio) lives
    on the drone, not in the yaml -- the yaml only carries the *effective*
    4-DOF parameters derived from that geometry.
    """

    def __init__(self, g: float, drone):
        super().__init__(g, drone.params)

        allocation = build_allocation_matrix(
            drone.rotor_positions_b_t,
            drone.rotor_axes_b_t,
            drone.directions.reshape(-1, drone.num_rotors)[0],
            drone.rotor_drag_ratio,
        )
        rank = int(torch.linalg.matrix_rank(allocation))
        if rank < 6:
            raise ValueError(
                f"{type(drone).__name__} is not fully actuated: its allocation matrix has "
                f"rank {rank}, not 6. All rotor axes parallel? Use LeePositionController."
            )
        self.allocation = nn.Parameter(allocation)
        self.allocation_pinv = nn.Parameter(torch.linalg.pinv(allocation))

        inertia = drone.params["inertia"]
        self.inertia_diag = nn.Parameter(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]])
        )
        max_lateral = getattr(drone, "max_lateral_force_n", None)
        self.max_lateral_force = nn.Parameter(
            torch.tensor(float("inf") if max_lateral is None else float(max_lateral))
        )
        self.requires_grad_(False)

    def _compute(self, root_state, target_pos, target_vel, target_acc, target_yaw, body_rate):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            ang_vel = quat_rotate_inverse(rot, ang_vel)

        # Identical position law to LeePositionController, up to the point where
        # it would collapse `acc` onto a single body axis.
        acc = (
            (pos - target_pos) * self.pos_gain
            + (vel - target_vel) * self.vel_gain
            - self.g
            - target_acc
        )
        force_w = -self.mass * acc

        lateral = force_w[:, :2]
        lateral_norm = torch.norm(lateral, dim=-1, keepdim=True)
        force_w = torch.cat(
            [lateral * (self.max_lateral_force / lateral_norm.clamp_min(1e-6)).clamp(max=1.0),
             force_w[:, 2:]],
            dim=-1,
        )

        R = quaternion_to_rotation_matrix(rot)
        force_b = torch.einsum("bji,bj->bi", R, force_w)

        # Attitude target is independent of the force target -- that is the
        # whole point. Level, yawed to target_yaw.
        zeros = torch.zeros_like(target_yaw)
        b1_des = torch.cat([torch.cos(target_yaw), torch.sin(target_yaw), zeros], dim=-1)
        b3_des = torch.cat([zeros, zeros, torch.ones_like(target_yaw)], dim=-1)
        b2_des = torch.cross(b3_des, b1_des, dim=-1)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1],
            ang_error_matrix[:, 0, 2],
            ang_error_matrix[:, 1, 0],
        ], dim=-1)
        ang_acc = -ang_error * self.attitute_gain - ang_vel * self.ang_rate_gain
        torque_b = self.inertia_diag * ang_acc

        wrench_b = torch.cat([force_b, torque_b], dim=-1)
        thrusts = (self.allocation_pinv @ wrench_b.T).T
        return (thrusts / self.max_thrusts) * 2 - 1


def make_position_controller(drone, g: float = 9.81, mode: str = "auto"):
    """``fa`` for 6-DOF allocation, ``lee`` for the 4-DOF one, ``auto`` asks the drone.

    ``auto`` is what every caller should use; the explicit values exist so the
    same airframe can be flown both ways for comparison.
    """
    if mode not in ("auto", "fa", "lee"):
        raise ValueError(f"mode must be auto/fa/lee, got {mode!r}")
    if mode == "fa" or (mode == "auto" and getattr(drone, "fully_actuated", False)):
        return FullyActuatedPositionController(g, drone)
    return LeePositionController(g, drone.params)
