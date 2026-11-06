"""Freeze an aerial manipulator's arm dynamically while keeping it statically.

The problem this exists for: a drone controller tuned for a fixed airframe
diverges once the arm starts moving. Every joint that accelerates throws a
reaction wrench into ``base_link`` through the shared articulation -- Coriolis,
centrifugal and inertial terms the Lee controller has no model of and no
integral term to absorb. Hand-tuning gains that survive arbitrary arm motion
is a genuinely hard control problem, not a tuning pass.

``freeze_arm_dynamics`` sidesteps it the way humanoid pipelines sidestep
whole-body dynamics in the retargeting stage: make the plant the simple one
the controller was designed for, get a clean kinematic reference out of it,
and leave the real coupled dynamics to a learned controller later. Concretely:

* every arm link keeps only a small ``residual`` fraction of its real mass and
  inertia, so moving it transmits only that fraction of the real reaction
  wrench into the drone body;
* the other ``1 - residual`` of the arm's mass, plus its rotational inertia
  about ``base_link``'s origin, is added onto ``base_link`` -- so the drone
  still carries the arm's full weight and still has the arm's swing inertia to
  rotate.

``residual`` is a fraction rather than a flat epsilon on purpose, and its
response is NOT monotonic -- freezing harder is not safer. Measured on
ShelfBucketGrasp (GPU pipeline) under an identical hard arm sweep, worst body
tilt / worst position error over 500 steps:

    residual   1.0      0.3      0.1      0.02     0.005     0.0005
    tilt       7.4 deg  2.4 deg  1.3 deg  9.9 deg  84.0 deg  95.1 deg
    pos err    0.17 m   0.046 m  0.029 m  0.26 m   2.9 m     4.5 m

Below ~0.1 the mass ratio across the articulation (3.4 kg base vs a 0.3 g link
at 0.02, a 7 mg link at 0.0005) puts PhysX's GPU articulation solver into a
regime where it is worse than no freezing at all, and by 0.005 it is divergent.
``DEFAULT_RESIDUAL`` sits at the measured optimum. A flat epsilon mass, which
is the obvious first implementation, lands squarely in the divergent end of
that table.

Also measured and rejected: leaving the grasping links (``gripper``/``jaw``) at
full mass while freezing the rest. The motivation was real -- a 0.00024 kg jaw
closing on a 0.6 kg bucket is a 2500:1 ratio where the unfrozen jaw is a benign
50:1 -- but it made flight markedly worse (20.9 deg tilt vs 9.9 at the same
residual), because those are the longest-lever-arm links. If contact-rich
grasping later needs heavier fingers, re-measure rather than reinstating it
from the argument alone.

The arm therefore still looks right, still poses right, still collides, and
still costs the same hover thrust -- it just stops kicking back.

Two deliberate approximations, both stated rather than hidden:

1. Each arm link is lumped as a POINT MASS at its link origin (parallel-axis
   term only; the link's own inertia tensor about its own centre is dropped).
   For thin links at ~0.2m lever arms that is worth ~10% of the arm's own
   contribution, and the arm is ~20% of the whole airframe's inertia.
2. The arm's mass is added at ``base_link``'s existing centre of mass rather
   than shifting the combined CoM. That keeps ``base_link``'s CoM on its prim
   origin, which is what ``AmovDrone.apply_action`` assumes when it takes the
   aggregated rotor thrust's pitch/roll torque about
   ``base_link.get_world_poses()``. It also means the arm's static weight
   torque disappears, so the drone hovers level instead of trimmed -- for this
   mode that is the point, not a loss.

Neither approximation touches the CONTROLLER's model: ``LeePositionController``
reads mass/inertia from the drone's yaml, not from the live articulation.
Freezing only changes the plant, and it changes it toward the rigid airframe
that yaml already describes.
"""

import logging
from typing import Optional

import torch

from omni_drones.utils.torch import quaternion_to_rotation_matrix

#: Fraction of its real mass/inertia each arm link keeps. The measured optimum
#: -- see the module docstring's table; both larger AND smaller are worse.
DEFAULT_RESIDUAL = 0.1


def _arm_body_indices(body_names):
    """Every body that is neither the drone's own ``base_link`` nor a rotor.

    Covers both arm layouts this project has: a runtime-mounted manipulator
    (iris650_soarm100 -> base/shoulder/upper_arm/lower_arm/wrist/gripper/jaw)
    and one baked into the drone's own USD (amov_drone_arm -> link1..6 and the
    gripper fingers). Rotor bodies are named ``rotor_0..3`` / ``rotor_link_1..4``
    on every drone here, hence the prefix test.
    """
    return [
        i for i, n in enumerate(body_names)
        if n != "base_link" and not n.startswith("rotor")
    ]


def freeze_arm_dynamics(am, residual: float = DEFAULT_RESIDUAL) -> Optional[dict]:
    """Move most of the arm's mass/inertia onto ``base_link``, leaving the arm
    links with only ``residual`` of theirs. Call once, after ``initialize()``.

    Args:
        am: an ``AerialManipulatorBase`` (needs its live ``ArticulationView``).
        residual: fraction of its real mass/inertia each arm link keeps, and
            therefore roughly the fraction of the real reaction wrench the
            drone still feels. 1.0 is plain coupled physics.

    Returns:
        A summary dict, or ``None`` if this robot has no arm bodies at all.
    """
    view = am.drone._view
    body_names = list(view.body_names)
    arm_idx = _arm_body_indices(body_names)
    if not arm_idx:
        return None
    base_i = body_names.index("base_link")

    masses = view.get_body_masses().reshape(len(body_names))
    inertias = view.get_body_inertias().reshape(len(body_names), 9)

    # Each arm link's position relative to base_link, expressed in base_link's
    # own frame. Positions come from the physics view (no quaternion-convention
    # ambiguity); the single rotation needed is base_link's own, taken from the
    # drone class's get_world_poses(), which is this repo's (w,x,y,z).
    link_pos = view._physics_view.get_link_transforms().reshape(-1, len(body_names), 7)[0, :, :3]
    base_pos, base_rot = am.drone.get_world_poses()
    base_pos = base_pos.reshape(-1, 3)[0].to(link_pos.device)
    R = quaternion_to_rotation_matrix(base_rot.reshape(-1, 4)[:1])[0].to(link_pos.device)
    rel = (link_pos - base_pos) @ R  # world -> base frame (R^T v == v @ R)

    # Link-local CoM offsets are ignored by the point-mass lumping above; say
    # so out loud if any of them is big enough to matter for a given asset.
    com_local = view.get_body_coms()[0].reshape(-1, len(body_names), 3)[0]
    worst_com = float(com_local[arm_idx].norm(dim=-1).max()) if len(arm_idx) else 0.0
    if worst_com > 0.02:
        logging.warning(
            "freeze_arm_dynamics: an arm link's local CoM sits %.3fm off its prim "
            "origin; the point-mass lumping uses link origins and ignores that "
            "offset. Add the rotated CoM offset here if this asset needs it.",
            worst_com,
        )

    # Only the moved share (1 - residual) goes onto base_link, so total mass
    # and total swing inertia stay exactly what they were.
    moved = 1.0 - residual
    arm_mass = float(masses[arm_idx].sum())
    inertia_add = torch.zeros(3, 3, device=link_pos.device, dtype=torch.float32)
    eye = torch.eye(3, device=link_pos.device, dtype=torch.float32)
    for i in arm_idx:
        d = rel[i].float()
        inertia_add += moved * float(masses[i]) * (d.dot(d) * eye - torch.outer(d, d))

    new_masses = masses.clone().float()
    new_masses[arm_idx] = masses[arm_idx].float() * residual
    new_masses[base_i] = float(masses[base_i]) + moved * arm_mass

    new_inertias = inertias.clone().float()
    new_inertias[arm_idx] = inertias[arm_idx].float() * residual
    new_inertias[base_i] = (
        inertias[base_i].reshape(3, 3).float() + inertia_add.to(inertias.device)
    ).reshape(9)

    count = view.count
    view.set_body_masses(new_masses.unsqueeze(0).expand(count, -1).contiguous())
    view.set_body_inertias(new_inertias.unsqueeze(0).expand(count, -1, -1).contiguous())

    # Read back: set_body_masses/set_body_inertias go through the physics
    # backend, and a write that does not land (wrong pipeline, stale buffer)
    # is otherwise indistinguishable from one that does -- except that the
    # plant is then a half-frozen arm, which is worse than either end state.
    got_m = view.get_body_masses().reshape(len(body_names)).float()
    got_i = view.get_body_inertias().reshape(len(body_names), 9).float()
    bad_m = float((got_m - new_masses.to(got_m.device)).abs().max())
    # PhysX stores inertia as principal moments plus a CoM orientation, so a
    # tensor written with off-diagonal terms reads back diagonalised. Compare
    # the eigenvalues, which that representation change leaves alone.
    want_eig = torch.linalg.eigvalsh(new_inertias.reshape(-1, 3, 3).double()).float()
    got_eig = torch.linalg.eigvalsh(got_i.reshape(-1, 3, 3).double()).float()
    bad_i = float((got_eig - want_eig.to(got_eig.device)).abs().max())
    if bad_m > 1e-3 or bad_i > 1e-3:
        raise RuntimeError(
            f"freeze_arm_dynamics: the mass/inertia write did not land "
            f"(max mass error {bad_m:.4g} kg, max inertia error {bad_i:.4g}). "
            f"Leaving the articulation half-frozen would be worse than either "
            f"end state, so this is fatal.\n"
            + "\n".join(
                f"  {body_names[i]}: mass {float(got_m[i]):.6f}/{float(new_masses[i]):.6f} "
                f"inertia_diag {[round(float(got_i[i][k]), 6) for k in (0, 4, 8)]}"
                f"/{[round(float(new_inertias[i][k]), 6) for k in (0, 4, 8)]}"
                for i in range(len(body_names))
            )
        )

    # Deliberately NOT scaling the arm's joint drive gains by the residual.
    # The idea is tempting -- a PD drive's closed-loop frequency is sqrt(kp/I),
    # so making a link 1000x lighter and leaving kp alone multiplies it by ~31,
    # past what a 100 Hz step can integrate. It was tried, to fix
    # amov950_airbot_dexterous_bottom's flailing Inspire hand, and made that
    # robot far worse: the sweep threw the drone 19 m up at 154 deg of tilt,
    # against merely going non-finite before. With the fingers' authored
    # kp=0.05..0.07 scaled down they have no drive left at all, and
    # self-collision between them does the rest. That asset's gains are
    # unusable in both directions, which is why its preset declares
    # arm_write="teleport" instead.
    summary = {
        "arm_bodies": [body_names[i] for i in arm_idx],
        "residual": residual,
        "arm_mass": arm_mass,
        "base_mass": float(new_masses[base_i]),
        "inertia_added_diag": [float(v) for v in inertia_add.diagonal()],
        "base_inertia_diag": [float(new_inertias[base_i][k]) for k in (0, 4, 8)],
    }
    logging.info("freeze_arm_dynamics: %s", summary)
    return summary
