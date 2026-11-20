"""Hummingbird, 340mm class -- OmniDrones' own asset and rotor model.

Source: ``omni_drones/robots/assets/usd/hummingbird.usd`` +
``hummingbird.yaml`` (this repo). ``arm_lengths[0] = 0.17`` in that yaml ->
~340mm rotor-to-rotor diagonal. Flown by ``AerialCabinetGrasp`` (as
``hummingbird340_2dof_gripper``, the robot that reaches into the cabinet).

The asset and every parameter are OmniDrones' own; the one thing replaced is
*where the rotor thrust is applied*. ``MultirotorBase.apply_action`` pushes
each rotor body separately, through a ``RigidPrimView`` over the four
``rotor_*`` links. Those are child links of an articulation, and on the GPU
pipeline -- which every task env here runs on -- PhysX silently ignores a
rigid-body force write to them: the drone is commanded to hover and falls at
exactly g (measured in ``AerialCabinetGrasp``: 0.85 m in the first 0.4 s, rotor
commands rising the whole way down). It never showed before because the only
thing that had flown this robot was ``examples/show_am_action.py``, which runs
on the CPU pipeline, where the same write lands.

So the wrench is aggregated onto ``base_link`` and applied through the
articulation view, the same thing ``AmovDrone`` and ``RotorlessMultirotor`` do
for the same reason. The thrust and drag-torque values are still
``RotorGroup``'s; only the lever arms come from the yaml's ``arm_lengths`` /
``rotor_angles`` instead of the rotor bodies' live poses, which for rotors
rigidly fixed to the airframe is the same thing.
"""

import torch

from omni_drones.robots.drone.hummingbird import Hummingbird as _Hummingbird
from omni_drones.utils.torch import quat_axis, quat_rotate

from .drone_base import DroneSpec, register_drone

#: Height of the rotor hubs above ``base_link``'s origin, metres (the asset's
#: ``rotor_*`` bodies sit at z = 0.01).
ROTOR_HEIGHT_M = 0.01


class _Hummingbird340(_Hummingbird):
    """``Hummingbird`` with its rotor wrench applied to ``base_link``."""

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        super().initialize(prim_paths_expr=prim_paths_expr, track_contact_forces=track_contact_forces)
        rotor_cfg = self.params["rotor_configuration"]
        arm = torch.as_tensor(rotor_cfg["arm_lengths"], dtype=torch.float32, device=self.device)
        angle = torch.as_tensor(rotor_cfg["rotor_angles"], dtype=torch.float32, device=self.device)
        self.rotor_positions_b_t = torch.stack(
            [arm * torch.cos(angle), arm * torch.sin(angle), torch.full_like(arm, ROTOR_HEIGHT_M)], dim=-1)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = self.rotors(rotor_cmds, self.rotor_params)

        rot = self.rot.reshape(-1, 4)
        n = rot.shape[0]
        up_w = quat_axis(rot, axis=2).reshape(*self.shape, 1, 3)
        quat = rot.unsqueeze(1).expand(n, self.num_rotors, 4).reshape(-1, 4)
        rel_w = quat_rotate(
            quat, self.rotor_positions_b_t.expand(n, self.num_rotors, 3).reshape(-1, 3)
        ).reshape(*self.shape, self.num_rotors, 3)

        thrust_w = thrusts.unsqueeze(-1) * up_w
        self.thrusts[..., 2] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * up_w).sum(-2)
        if self.rotor_joint_indices is not None:
            # Visual only: the propellers turn at the commanded rate.
            rot_vel = self.throttle * self.directions * self.MAX_ROT_VEL
            self._view.set_joint_velocities(
                rot_vel.reshape(-1, self.num_rotors), joint_indices=self.rotor_joint_indices)

        self.forces.zero_()
        self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]

        net_thrust = thrust_w.sum(dim=-2)
        pitch_roll = torch.cross(rel_w, thrust_w, dim=-1).sum(dim=-2)
        # Through the articulation view, which writes to the root link -- not
        # through self.base_link, a RigidPrimView on a link of an articulation.
        self._view.apply_forces_and_torques_at_pos(
            (self.forces + net_thrust).reshape(-1, 3),
            (self.torques + pitch_roll).reshape(-1, 3),
            is_global=True,
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)


OMNIDRONES_340 = register_drone(
    DroneSpec(
        name="omnidrones_340",
        drone_class=_Hummingbird340,
        usd_path=_Hummingbird.usd_path,
        param_path=_Hummingbird.param_path,
        mount_body_name="base_link",
        class_size_mm=340,
        livery=False,  # the asset this project's livery copies its colours from
        note=(
            "OmniDrones' asset and rotor model; thrust aggregated onto base_link so it "
            "also flies on the GPU pipeline (see the module docstring)."
        ),
    )
)
