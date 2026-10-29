"""AM-bench's FAHexa arm (4 DOF + 2-finger gripper), as shipped (not re-mounted).

Source: baked directly into ``ambench_750``'s own USD
(``/home/benhua/ambench/source/ambench/ambench/assets/robots/hexa_scorpion.usd``)
-- same ``bundled_into_drone=True`` situation as ``ambench_quad_arm``, no
separate arm USD to mount. Unlike the quad variant, this arm keeps its
original orientation ("六旋翼保持原上置装配" -- already top-mounted on the
source hexacopter, no re-mount step applied). Joint names/ranges below
cross-checked against ``assets/mjcf/ambench_fa_hexa_750_final/robot.xml``'s
``conversion_report.json``, exported from this same USD.

Chain: ``arm_base_link`` -> ``arm_link1_pitch_link`` (``arm_link1_pitch_joint``,
range -0.4..3.0 rad) -> ``arm_link2_pitch_link`` (``arm_link2_pitch_joint``,
+/-10 rad) -> ``arm_link3_pitch_link`` (``arm_link3_pitch_joint``, +/-10 rad)
-> ``arm_link4_roll_link`` (``arm_link4_roll_joint``, +/-10 rad) ->
``ee_link`` -> ``manipulation_lfinger_link``/``manipulation_rfinger_link``
(slide joints ``arm_lfinger_joint``/``arm_rfinger_joint``, 0.001 to 0.045 m
-- note: same joint *names* as the quad arm but a different range).
Different joint axis layout from the quad arm (pitch/pitch/pitch/roll vs.
yaw/pitch/pitch/pitch) -- these are genuinely different arms, not the same
one reused, despite AM-bench's shared 4+2 DOF count.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH
from .manipulator_base import ManipulatorSpec, register_manipulator

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "ambench_fa_hexa_750_final", "robot.xml")
USD_PATH = "/home/benhua/ambench/source/ambench/ambench/assets/robots/hexa_scorpion.usd"  # == ambench_750's own USD

AMBENCH_HEXA_ARM = register_manipulator(
    ManipulatorSpec(
        name="ambench_hexa_arm",
        usd_path=USD_PATH,
        root_body_name="arm_base_link",
        eef_body_name="ee_link",
        arm_joint_names=(
            "arm_link1_pitch_joint",
            "arm_link2_pitch_joint",
            "arm_link3_pitch_joint",
            "arm_link4_roll_joint",
        ),
        gripper_bundled=True,
        bundled_into_drone=True,
        default_gripper="ambench_finger_gripper",
        note=(
            f"Pre-baked into ambench_750's real Isaac Sim USD at {USD_PATH}; "
            f"joint names/ranges cross-checked against {MJCF_SOURCE_PATH}."
        ),
    )
)
