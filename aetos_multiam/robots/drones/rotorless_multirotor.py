"""A multirotor whose asset has no rotor rigid bodies -- only rotor visuals.

AM-Bench authors ``quad_scorpion.usd``/``hexa_scorpion.usd`` that way: the
articulation's only bodies are ``base_link`` plus the arm chain, and the
propellers exist solely as geometry under ``base_link/visuals/...`` (AM-Bench
builds its own rotor wrench from hard-coded body-frame geometry too -- see
``ambench/robots/visuals/propeller_specs.py`` and
``ambench/controllers/utils/control_allocation.py``). OmniDrones'
``MultirotorBase`` instead reads each rotor's *live world pose* off a
``rotor_*`` ``RigidPrimView`` to build its thrust directions and lever arms,
so it has nothing to read on these assets.

This class supplies that geometry statically and aggregates the wrench onto
``base_link`` itself. Two separate reasons to aggregate rather than push each
rotor:

* there is no rotor body to push, and
* ``RigidPrimView.apply_forces_and_torques_at_pos`` on an articulation's child
  links is silently ignored by PhysX anyway -- the same finding that made
  ``AmovDrone.apply_action`` aggregate onto ``base_link``, whose structure this
  ``apply_action`` follows.

Per-rotor thrust axes are allowed to be non-parallel (``rotor_axes_b``), which
is how AM-Bench's FA-Hexa gets its fully-actuated wrench from six fixed canted
rotors. Such a platform flies correctly through ``LeePositionController``: that
controller commands only four DOFs (body-Z thrust and three torques), and the
4-row map from per-rotor thrust to those four is *exactly* representable in
the yaml's own parameter family -- solve ``arm_lengths``/``rotor_angles`` from
each rotor's real torque-per-thrust and fold the lever arm's own yaw component
into ``moment_constants``. Checked numerically against FA-Hexa's geometry:
residual 5.6e-17. What it gives up is not correctness but *capability* -- the
lateral force the cant could produce at a level attitude, which a 4-DOF
controller never asks for, so it tilts to translate like any underactuated
multirotor.

To actually use that capability, set ``fully_actuated`` and fly the drone
through ``controllers.drone.FullyActuatedPositionController`` instead, which
builds a 6xN allocation matrix from the same geometry declared here (see
``controllers/drone/allocation.py``). The plant below is unchanged either way
-- both controllers end up handing it the same per-rotor thrust vector.

Two conventions make that exactness work, and both matter:

``rotor_axes_b`` is scaled so each axis's **z-component is 1**, not to unit
length. The thrust variable everything else in OmniDrones carries (RotorGroup's
``KF * omega^2``, the controller's ``max_thrusts``, the mixer's row of ones) is
then each rotor's *vertical force component*, which is what lets the mixer's
hard-coded ``ones`` row stay correct on a canted platform. The real thrust
magnitude is ``T * |axis|`` and comes out of ``T * axis`` for free.

``rotor_drag_ratio`` carries the real rotor drag, because the yaml's
``moment_constants`` no longer can. The mixer gets yaw authority only through
``-direction * moment_constant / force_constant``, so a canted rotor's lever
arm contribution to yaw (``r x F``, which for a canted axis has a z-component)
has to be folded in there -- leaving the yaml's ratio no longer equal to the
physical drag ratio. On a plain +Z quad the two coincide (``r x z_hat`` is
purely horizontal) and nothing changes.
"""

import torch
from torch.func import vmap

from omni_drones.robots import RobotBase
from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.utils.torch import quat_rotate


class RotorlessMultirotor(MultirotorBase):
    """``MultirotorBase`` for an asset with visual-only rotors.

    Subclasses declare the geometry ``MultirotorBase`` would otherwise have
    read from the stage:

    ``rotor_positions_b``
        ``(num_rotors, 3)`` rotor origins in the ``base_link`` frame, metres.
    ``rotor_axes_b``
        ``(num_rotors, 3)`` thrust directions in the ``base_link`` frame, each
        scaled so its z-component is 1 (see the module docstring). Defaults to
        ``+Z`` for every rotor.
    ``rotor_drag_ratio``
        Reaction torque per unit rotor thrust, about the rotor's own axis
        (AM-Bench's ``reaction_torque_ratio``). Not read from the yaml -- see
        the module docstring.
    ``fully_actuated``
        Set when the canted axes span a rank-6 wrench space, so
        ``make_position_controller`` picks the 6-DOF allocator over the
        4-DOF mixer. Checked for real at controller construction, not trusted.
    ``max_lateral_force_n``
        Worst-azimuth horizontal force the platform can hold at hover without
        any rotor leaving ``[0, max_thrust]`` -- the limit the 6-DOF
        controller clamps its lateral command to. ``None`` = no clamp.
    """

    rotor_positions_b: tuple = ()
    rotor_axes_b: tuple = ()
    rotor_drag_ratio: float = 0.02
    fully_actuated: bool = False
    max_lateral_force_n: float = None

    def _articulation_properties_prim_path(self, spawn_root_prim_path: str) -> str:
        # ArticulationRootAPI is on base_link, so the solver settings belong
        # there too -- RobotBase.spawn would otherwise author them on the spawn
        # prim and fail outright ("No articulation schema present").
        return f"{spawn_root_prim_path}/base_link"

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        # These assets carry ArticulationRootAPI on base_link, not on the spawn
        # prim, so the ArticulationView has to be created there -- the same
        # resolution AmovDrone does for its URDF-imported chain. base_link then
        # IS that view (wrapping a separate RigidPrimView around an
        # articulation's root link is what AmovDrone's own comment warns makes
        # get_world_poses read out of bounds).
        root_expr = prim_paths_expr or f"/World/envs/.*/{self.name}_.*"
        RobotBase.initialize(self, prim_paths_expr=f"{root_expr}/base_link")
        self.base_link = self._view
        self.prim_paths_expr = root_expr
        self.rotors_view = None
        self.rotor_joint_indices = None  # rotors are visuals: no DOFs to spin
        self._setup_rotor_model_and_buffers()

        positions = torch.as_tensor(self.rotor_positions_b, dtype=torch.float32, device=self.device)
        if positions.shape != (self.num_rotors, 3):
            raise ValueError(
                f"{type(self).__name__}.rotor_positions_b must be "
                f"({self.num_rotors}, 3), got {tuple(positions.shape)} -- it has to agree "
                f"with the yaml's rotor_configuration.num_rotors."
            )
        axes = (
            torch.as_tensor(self.rotor_axes_b, dtype=torch.float32, device=self.device)
            if len(self.rotor_axes_b)
            else torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_rotors, 3)
        )
        if axes.shape != (self.num_rotors, 3):
            raise ValueError(
                f"{type(self).__name__}.rotor_axes_b must be ({self.num_rotors}, 3), "
                f"got {tuple(axes.shape)}."
            )
        if not torch.allclose(axes[:, 2], torch.ones_like(axes[:, 2]), atol=1e-4):
            raise ValueError(
                f"{type(self).__name__}.rotor_axes_b must be scaled so every axis has "
                f"z-component 1, got {axes[:, 2].tolist()} -- the thrust variable is the "
                "vertical component, which is what keeps the controller's mixer exact "
                "on a canted platform (see the module docstring)."
            )
        self.rotor_positions_b_t = positions
        self.rotor_axes_b_t = axes

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        # Rotate the static body-frame geometry into world frame with base_link's
        # own orientation -- the one thing MultirotorBase got from rotor poses.
        rot = self.rot.reshape(-1, 4)
        n = rot.shape[0]
        quat = rot.unsqueeze(1).expand(n, self.num_rotors, 4).reshape(-1, 4)
        axes_w = quat_rotate(quat, self.rotor_axes_b_t.expand(n, self.num_rotors, 3).reshape(-1, 3))
        rel_w = quat_rotate(quat, self.rotor_positions_b_t.expand(n, self.num_rotors, 3).reshape(-1, 3))
        axes_w = axes_w.reshape(*self.shape, self.num_rotors, 3)
        rel_w = rel_w.reshape(*self.shape, self.num_rotors, 3)

        thrust_w = thrusts.unsqueeze(-1) * axes_w
        self.thrusts[..., 2] = thrusts
        # Rotor drag reacts about each rotor's own spin axis -- from
        # rotor_drag_ratio, NOT from the yaml's moment_constants, which on a
        # canted platform also carry the lever arm's yaw share (module docstring).
        # -directions: the sign convention RotorGroup.forward already applies to
        # its own moments (``(t * KM) * -self.directions``).
        drag = -self.directions * self.rotor_drag_ratio * thrusts
        self.torques[:] = (drag.unsqueeze(-1) * axes_w).sum(-2)

        self.forces.zero_()
        self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]

        net_thrust = thrust_w.sum(dim=-2)
        # Lever arms are already relative to base_link's origin, so unlike
        # AmovDrone there is no world-vs-local reference-frame trap here.
        pitch_roll = torch.cross(rel_w, thrust_w, dim=-1).sum(dim=-2)

        self.base_link.apply_forces_and_torques_at_pos(
            (self.forces + net_thrust).reshape(-1, 3),
            (self.torques + pitch_roll).reshape(-1, 3),
            is_global=True,
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)
