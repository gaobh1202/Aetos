"""AM-bench UAQuad, 600mm class -- real Isaac Sim USD (AM-bench is itself Isaac-Sim-based).

Source: ``/home/benhua/ambench/source/ambench/ambench/assets/robots/quad_scorpion.usd``
(22MB, self-contained) -- AM-bench is built on IsaacLab/Isaac Sim, so unlike
the MJCF-only components elsewhere in this project, this asset is *already*
a native Isaac Sim USD. No conversion step is needed for geometry/joints.

This USD bundles the arm and gripper into the same file (see
``manipulators/ambench_quad_arm.py``'s ``bundled_into_drone=True``) --
AM-bench authored drone+arm+gripper as one combined articulation, not
separate mountable pieces. Exact mass/rotor geometry (see
``conversion_report.json`` cross-checked while this project pulled in the
MJCF viewer export): total mass **4.40692603122443 kg**, rotor positions
(m, body frame) ``[(0.21213,0.21213,0.05), (-0.21213,0.21213,0.05),
(-0.21213,-0.21213,0.05), (0.21213,-0.21213,0.05)]`` (arm length 0.3 m on
all 4, X-config), spin directions ``[-1, 1, -1, 1]``, thrust axis +Z,
thrust range 0-23 N per rotor.

Still can't build a ``RotorGroup``-compatible yaml: AM-bench models rotor
thrust with its own reduced-order ``RotorActuator`` (direct thrust command
with a response-time-constant transient, no physical rotor speed), not the
``force_constant``/``max_rotation_velocity`` pair ``RotorGroup`` expects.
That pair would need to be *chosen* (e.g. back-solved from a target max
rotor speed), not read off -- not fabricated here. This is the one
remaining real gap; the USD itself is not a gap any more.
"""

import os.path as osp

from omni_drones.robots.drone.multirotor import MultirotorBase

from aetos_multiam.assets import ASSET_PATH, USD_PATH
from .drone_base import DroneSpec, register_drone

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "ambench_quad_600_top_arm_final", "robot.xml")
AMBENCH_600_USD_PATH = (
    "/home/benhua/ambench/source/ambench/ambench/assets/robots/quad_scorpion.usd"
)
AMBENCH_600_PARAM_PATH = osp.join(USD_PATH, "ambench_600.yaml")  # does not exist yet (needs KF/KM, see above)

TOTAL_MASS_KG = 4.40692603122443
ROTOR_POSITIONS_M = (
    (0.2121320366859436, 0.2121320366859436, 0.05),
    (-0.2121320366859436, 0.2121320366859436, 0.05),
    (-0.2121320366859436, -0.2121320366859436, 0.05),
    (0.2121320366859436, -0.2121320366859436, 0.05),
)
SPIN_DIRECTIONS = (-1, 1, -1, 1)
MAX_THRUST_PER_ROTOR_N = 23.0


class _AmbenchQuadScorpion(MultirotorBase):
    usd_path: str = AMBENCH_600_USD_PATH
    param_path: str = AMBENCH_600_PARAM_PATH


AMBENCH_600 = register_drone(
    DroneSpec(
        name="ambench_600",
        drone_class=_AmbenchQuadScorpion,
        usd_path=AMBENCH_600_USD_PATH,
        param_path=AMBENCH_600_PARAM_PATH,
        mount_body_name="base_link",
        class_size_mm=600,
        note=(
            f"Real Isaac Sim USD at {AMBENCH_600_USD_PATH} (AM-bench's own "
            f"asset, arm+gripper bundled in -- see {MJCF_SOURCE_PATH} for a "
            "cross-checked geometry reference); still no RotorGroup-compatible "
            "KF/KM (AM-bench's own rotor model doesn't expose that pair)."
        ),
    )
)
