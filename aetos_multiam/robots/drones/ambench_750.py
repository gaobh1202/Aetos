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

Same gap as ``ambench_600``: no ``RotorGroup``-compatible KF/KM (AM-bench's
reduced-order ``RotorActuator`` has no equivalent pair to port). A
tilt-rotor layout also needs ``RotorGroup``'s per-rotor thrust-axis support
(not just angle+arm-length), which isn't verified to exist in the version
this benchmark imports.
"""

import os.path as osp

from omni_drones.robots.drone.multirotor import MultirotorBase

from aetos_multiam.assets import ASSET_PATH, USD_PATH
from .drone_base import DroneSpec, register_drone

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "ambench_fa_hexa_750_final", "robot.xml")
AMBENCH_750_USD_PATH = (
    "/home/benhua/ambench/source/ambench/ambench/assets/robots/hexa_scorpion.usd"
)
AMBENCH_750_PARAM_PATH = osp.join(USD_PATH, "ambench_750.yaml")  # does not exist yet (needs KF/KM, see above)

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
SPIN_DIRECTIONS = (1, -1, 1, -1, 1, -1)
MAX_THRUST_PER_ROTOR_N = 23.0
CLAMPED_INERTIA_ZZ = 0.395604  # see docstring; source Izz=0.460 violates the triangle inequality


class _AmbenchHexaScorpion(MultirotorBase):
    usd_path: str = AMBENCH_750_USD_PATH
    param_path: str = AMBENCH_750_PARAM_PATH


AMBENCH_750 = register_drone(
    DroneSpec(
        name="ambench_750",
        drone_class=_AmbenchHexaScorpion,
        usd_path=AMBENCH_750_USD_PATH,
        param_path=AMBENCH_750_PARAM_PATH,
        mount_body_name="base_link",
        class_size_mm=750,
        note=(
            f"Real Isaac Sim USD at {AMBENCH_750_USD_PATH} (AM-bench's own "
            f"asset, arm+gripper bundled in -- see {MJCF_SOURCE_PATH} for a "
            "cross-checked geometry reference); still no RotorGroup-compatible KF/KM."
        ),
    )
)
