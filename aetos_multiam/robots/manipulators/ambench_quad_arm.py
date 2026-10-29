"""AM-bench's UAQuad arm (4 DOF + 2-finger gripper), top-mounted variant.

Source: baked directly into ``ambench_600``'s own USD
(``/home/benhua/ambench/source/ambench/ambench/assets/robots/quad_scorpion.usd``)
-- AM-bench authored drone+arm+gripper as one combined articulation, so
there's no separate arm USD to mount (``bundled_into_drone=True``, see
``manipulator_base.py``'s docstring). Joint names/transform below were
cross-checked against this project's MJCF viewer export
(``assets/mjcf/ambench_quad_600_top_arm_final/robot.xml`` and its
``conversion_report.json``), which was itself exported from this same USD.

Chain: ``arm_base_link`` (mount, at body-frame (0.1, 0, 0.003) m, quat wxyz
(0,1,0,0) -- i.e. flipped 180 deg about X to face up) -> ``arm_link1_yaw_link``
(``arm_link1_yaw_joint``, range +/-90 deg) -> ``arm_link2_pitch_link``
(``arm_link2_pitch_joint``, +/-90 deg) -> ``arm_link3_pitch_link``
(``arm_link3_pitch_joint``, +/-90 deg) -> ``arm_link4_pitch_link``
(``arm_link4_pitch_joint``, +/-90 deg) -> ``ee_link`` ->
``manipulation_lfinger_link``/``manipulation_rfinger_link`` (slide joints
``arm_lfinger_joint``/``arm_rfinger_joint``, 0 to 0.05 m). All 4 arm joints:
kp=100000, kv=1000, torque limit 1000 N*m (Isaac Lab's original SI gains,
per the source's own notes). Initial ``ee_link`` pose relative to the drone
body: (0.27660, 0.01110, 0.23950) m, facing forward.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH
from .manipulator_base import ManipulatorSpec, register_manipulator

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "ambench_quad_600_top_arm_final", "robot.xml")
USD_PATH = "/home/benhua/ambench/source/ambench/ambench/assets/robots/quad_scorpion.usd"  # == ambench_600's own USD

AMBENCH_QUAD_ARM = register_manipulator(
    ManipulatorSpec(
        name="ambench_quad_arm",
        usd_path=USD_PATH,
        root_body_name="arm_base_link",
        eef_body_name="ee_link",
        arm_joint_names=(
            "arm_link1_yaw_joint",
            "arm_link2_pitch_joint",
            "arm_link3_pitch_joint",
            "arm_link4_pitch_joint",
        ),
        gripper_bundled=True,
        bundled_into_drone=True,
        default_gripper="ambench_finger_gripper",
        mount_translation=(0.1, 0.0, 0.003),
        mount_orientation=(0.0, 1.0, 0.0, 0.0),  # 180 deg about X -- confirmed, from the source's own robot.xml
        note=(
            f"Pre-baked into ambench_600's real Isaac Sim USD at {USD_PATH}; "
            f"joint names/transform cross-checked against {MJCF_SOURCE_PATH}."
        ),
    )
)
