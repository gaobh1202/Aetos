"""Fixed-tilt 6-DOF control allocation -- pure torch, no Isaac Sim import.

A multirotor whose rotors are *not* all parallel can produce an arbitrary
body wrench, not just thrust-along-body-z plus three torques. AM-Bench does
this in ``ambench/controllers/utils/control_allocation.py``: build a 6xN
matrix whose column ``i`` is the wrench rotor ``i`` contributes per unit
thrust, then ``pinv`` it and hit the desired wrench with it. Its
``fully_actuated=True`` branch is exactly that; its ``fully_actuated=False``
branch throws away the two lateral force rows, which is the underactuated
platform ``LeePositionController`` assumes.

This module is the same construction with one difference that matters: the
column is built from **this plant's own** rotor convention rather than from
a re-derived one, so allocation and simulation cannot disagree about a sign.
``RotorlessMultirotor.apply_action`` applies, per rotor:

    force  = T_i * axis_i
    torque = cross(r_i, T_i * axis_i)  +  (-dir_i * drag_ratio * T_i) * axis_i

so the column is ``[axis_i ; cross(r_i, axis_i) - dir_i * drag_ratio *
axis_i]`` and nothing else. ``dir_i`` is the yaml's ``directions`` entry (the
*negated* spin direction -- that sign lives in ``RotorGroup`` and is easy to
get backwards if you re-derive it from the asset instead).

``T_i`` is the rotor's **vertical force component**, not its thrust
magnitude, because ``axis_i`` is z-normalized rather than unit (see
``robots/drones/rotorless_multirotor.py``). That is the same variable the
yaml's ``force_constants`` and the controller's ``max_thrusts`` carry, so no
rescaling is needed anywhere.
"""

import math

import torch

__all__ = ["build_allocation_matrix", "lateral_force_envelope"]


def build_allocation_matrix(
    rotor_positions_b: torch.Tensor,
    rotor_axes_b: torch.Tensor,
    directions: torch.Tensor,
    drag_ratio: float,
) -> torch.Tensor:
    """``(6, num_rotors)`` map from per-rotor thrust to body wrench.

    Rows are ``[fx, fy, fz, tx, ty, tz]``. Args are ``(num_rotors, 3)``,
    ``(num_rotors, 3)``, ``(num_rotors,)`` and a scalar; ``directions`` is the
    yaml's own ``rotor_configuration.directions``.
    """
    pos = torch.as_tensor(rotor_positions_b, dtype=torch.float32)
    axes = torch.as_tensor(rotor_axes_b, dtype=torch.float32).to(pos.device)
    dirs = torch.as_tensor(directions, dtype=torch.float32).to(pos.device).reshape(-1)
    torque = torch.cross(pos, axes, dim=-1) - dirs.unsqueeze(-1) * drag_ratio * axes
    return torch.cat([axes, torque], dim=-1).transpose(-2, -1)


def lateral_force_envelope(
    allocation: torch.Tensor,
    weight_n: float,
    max_thrust: float,
    num_azimuths: int = 72,
) -> float:
    """Largest horizontal force holdable at hover from **every** azimuth, N.

    The honest limit for a level-attitude translation command: the allocation
    is exact only while every rotor stays inside ``[0, max_thrust]``, and a
    canted platform's lateral authority is not isotropic (FA-Hexa: 15.7 N at
    its worst azimuth vs. 20.0 N at its best). Takes the worst.
    """
    pinv = torch.linalg.pinv(allocation)
    worst = float("inf")
    for k in range(num_azimuths):
        cos_t, sin_t = math.cos(2.0 * math.pi * k / num_azimuths), math.sin(2.0 * math.pi * k / num_azimuths)
        lo, hi = 0.0, 10.0 * weight_n
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            wrench = torch.tensor(
                [mid * cos_t, mid * sin_t, weight_n, 0.0, 0.0, 0.0],
                dtype=allocation.dtype, device=allocation.device,
            )
            thrusts = pinv @ wrench
            if float(thrusts.min()) >= 0.0 and float(thrusts.max()) <= max_thrust:
                lo = mid
            else:
                hi = mid
        worst = min(worst, lo)
    return worst


if __name__ == "__main__":
    # Self-check on FA-Hexa's real geometry. Reads the constants straight out of
    # the drone module's source with `ast` rather than importing it: everything
    # under robots/drones/ transitively imports omni.isaac.core, which only
    # exists once SimulationApp has booted, and this check should not need a
    # simulator to run.
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[2] / "robots" / "drones" / "ambench_750.py"
    consts = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass

    # From ambench_750.yaml, which lives under assets/meshes/ and so isn't in git:
    # the yaml's own directions (negated spin), and max_rotation_velocities^2 *
    # force_constants.
    yaml_directions = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    max_thrust = 838.0 ** 2 * 2.836417e-05

    A = build_allocation_matrix(
        consts["ROTOR_POSITIONS_M"],
        consts["ROTOR_THRUST_AXES_Z_NORMALIZED"],
        yaml_directions,
        consts["ROTOR_REACTION_TORQUE_RATIO"],
    )
    assert torch.linalg.matrix_rank(A) == 6, "FA-Hexa must have full 6-DOF wrench authority"

    pinv = torch.linalg.pinv(A)
    weight = consts["TOTAL_MASS_KG"] * 9.81
    for wrench in [
        [0.0, 0.0, weight, 0.0, 0.0, 0.0],          # hover
        [10.0, 0.0, weight, 0.0, 0.0, 0.0],         # translate level, +x
        [0.0, -8.0, weight, 0.0, 0.0, 0.3],         # ...and yaw at the same time
        [0.0, 0.0, weight, 0.4, -0.4, 0.0],         # pure roll/pitch torque
    ]:
        w = torch.tensor(wrench)
        thrusts = pinv @ w
        # The allocation has to be EXACT, not least-squares: a rank-6 platform
        # can realize any wrench, so a residual here is a bug, not a limit.
        assert torch.allclose(A @ thrusts, w, atol=1e-4), f"{wrench} -> residual {A @ thrusts - w}"

    hover = pinv @ torch.tensor([0.0, 0.0, weight, 0.0, 0.0, 0.0])
    assert float(hover.min()) > 0.0 and float(hover.max()) < max_thrust, hover
    assert float(hover.max() - hover.min()) < 0.1, f"hover should be near-even, got {hover}"

    envelope = lateral_force_envelope(A, weight, max_thrust)
    # 15.69 N = 0.241 g, worst azimuth 120 deg -- the number ambench_750.py
    # declares as MAX_LATERAL_FORCE_N.
    assert 15.0 < envelope < 16.0, envelope
    print(f"ok -- rank 6, cond {torch.linalg.cond(A):.3f}, hover {hover[0]:.3f} N/rotor, "
          f"lateral envelope {envelope:.2f} N ({envelope / weight:.3f} g)")
