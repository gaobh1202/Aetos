"""AM-bench FAHexa, 750mm class -- real Isaac Sim USD (AM-bench is itself Isaac-Sim-based).

Source: ``/home/benhua/ambench/source/ambench/ambench/assets/robots/hexa_scorpion.usd``
(a thin ~2.8KB composition layer referencing sibling files under that same
directory's ``configuration/`` subfolder -- resolve it from its own
location, don't copy just this one file elsewhere). Like ``ambench_600``,
this is already a native Isaac Sim USD, no MJCF conversion needed; arm+
gripper are baked in (see ``manipulators/ambench_hexa_arm.py``'s
``bundled_into_drone=True``).

Exact mass/rotor geometry (cross-checked against ``conversion_report.json``
while this project pulled in the MJCF viewer export): total mass
**6.63499978557229 kg**, 6 *tilted* rotor thrust axes (genuine tilt-rotor
hex, not all +Z), spin directions ``[1, -1, 1, -1, 1, -1]``, max
rotor-center distance **0.750417 m**, thrust range 0-23 N per rotor. Source
body inertia ``(0.198, 0.198, 0.460)`` violates the triangle inequality as
authored -- the MJCF export's own conversion had to clamp Izz to
``0.999*(Ixx+Iyy) = 0.395604`` just to compile in MuJoCo; whether the live
USD's PhysX mass properties have the same issue (and need the same
clamping) hasn't been checked.

The cant is FIXED, not actuated -- six rotors at constant non-coplanar
angles, which is how this airframe gets full 6-DOF wrench authority with no
moving parts. Measured from that geometry: equal thrust gives net force
(0.003, 0, 5.196) and net torque ~0, so it hovers level on equal thrusts,
and the 6x6 wrench matrix has rank 6 / condition number 4.2.

``ambench_750.yaml`` now exists. ``RotorGroup`` never needed per-rotor
thrust-axis support -- ``RotorlessMultirotor`` carries the axes (and the
positions) itself, and the 4-row map LeePositionController actually commands
turns out to be *exactly* representable in the yaml's own parameter family
once the thrust variable is taken to be each rotor's vertical component
(residual 5.6e-17). See the yaml's header for the full derivation and
``rotorless_multirotor``'s docstring for the two conventions that make it
work.

That yaml path flies the airframe as an underactuated one. ``fully_actuated =
True`` below makes ``make_position_controller`` pick
``FullyActuatedPositionController`` instead, which uses the geometry here
directly in a 6xN allocation matrix and so can translate at a level attitude
-- up to ``MAX_LATERAL_FORCE_N``. Both controllers drive the same plant.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH, USD_PATH
from .drone_base import DroneSpec, register_drone
from .rotorless_multirotor import RotorlessMultirotor

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "ambench_fa_hexa_750_final", "robot.xml")
AMBENCH_750_USD_PATH = osp.join(USD_PATH, "ambench", "hexa_scorpion.usd")  # copied from AM-Bench (+ configuration/), see docs/ASSETS.md
AMBENCH_750_PARAM_PATH = osp.join(USD_PATH, "ambench_750.yaml")

TOTAL_MASS_KG = 6.63499978557229
ROTOR_POSITIONS_M = (
    (0.34175, 0.1585, -0.058),
    (-0.0335, 0.375, -0.058),
    (-0.30825, 0.2165, -0.058),
    (-0.30825, -0.2165, -0.058),
    (-0.0335, -0.375, -0.058),
    (0.34175, -0.1585, -0.058),
)
ROTOR_THRUST_AXES = (
    (-0.24941267034179734, 0.43335272628497695, 0.8660246731441039),
    (0.5003473468241608, 0.0, 0.8658247701042185),
    (-0.24941267034179734, -0.43335272628497695, 0.8660246731441039),
    (-0.24941267034179734, 0.43335272628497695, 0.8660246731441039),
    (0.5003473468241608, 0.0, 0.8658247701042185),
    (-0.24941267034179734, -0.43335272628497695, 0.8660246731441039),
)
# The same axes rescaled so each has z-component 1 -- the convention
# RotorlessMultirotor requires, because the thrust variable everything else
# carries is then the rotor's vertical force component (see that module).
ROTOR_THRUST_AXES_Z_NORMALIZED = (
    (-0.287997187697, 0.500393048516, 1.0),
    (0.577885230477, 0.0, 1.0),
    (-0.287997187697, -0.500393048516, 1.0),
    (-0.287997187697, 0.500393048516, 1.0),
    (0.577885230477, 0.0, 1.0),
    (-0.287997187697, -0.500393048516, 1.0),
)
SPIN_DIRECTIONS = (1, -1, 1, -1, 1, -1)
ROTOR_REACTION_TORQUE_RATIO = 0.02  # AM-bench's own reaction_torque_ratio
MAX_THRUST_PER_ROTOR_N = 23.0
# Worst-azimuth horizontal force holdable at hover with every rotor still
# inside [0, max_thrust] -- 0.241 g, at azimuth 120 deg; the best azimuth
# gives 20.0 N. Derived from the geometry above by
# controllers/drone/allocation.py's lateral_force_envelope(), which that
# module's own self-check re-derives and asserts.
MAX_LATERAL_FORCE_N = 15.69
CLAMPED_INERTIA_ZZ = 0.395604  # see docstring; source Izz=0.460 violates the triangle inequality


class _AmbenchHexaScorpion(RotorlessMultirotor):
    """Rotors are visuals only, and canted -- so both the positions and the
    per-rotor thrust axes live here rather than being read off ``rotor_*``
    rigid bodies. ``ambench_750.yaml``'s arm_lengths/rotor_angles/
    moment_constants are the *effective* values derived from this geometry;
    the physical drag ratio stays here, where the plant can use it."""

    usd_path: str = AMBENCH_750_USD_PATH
    param_path: str = AMBENCH_750_PARAM_PATH
    rotor_positions_b = ROTOR_POSITIONS_M
    rotor_axes_b = ROTOR_THRUST_AXES_Z_NORMALIZED
    rotor_drag_ratio = ROTOR_REACTION_TORQUE_RATIO
    fully_actuated = True
    max_lateral_force_n = MAX_LATERAL_FORCE_N


AMBENCH_750 = register_drone(
    DroneSpec(
        name="ambench_750",
        drone_class=_AmbenchHexaScorpion,
        usd_path=AMBENCH_750_USD_PATH,
        param_path=AMBENCH_750_PARAM_PATH,
        mount_body_name="base_link",
        class_size_mm=750,
        livery=False,  # already black-propped, kept as-is on request
        note=(
            f"Real Isaac Sim USD at {AMBENCH_750_USD_PATH} (AM-bench's own "
            f"asset, arm+gripper bundled in -- see {MJCF_SOURCE_PATH} for a "
            "cross-checked geometry reference). Flies via RotorlessMultirotor "
            "with effective mixer parameters derived from its fixed rotor cant "
            "(see ambench_750.yaml). Fully actuated: flown through "
            "FullyActuatedPositionController it translates without tilting, up "
            f"to {MAX_LATERAL_FORCE_N} N of lateral force."
        ),
    )
)
