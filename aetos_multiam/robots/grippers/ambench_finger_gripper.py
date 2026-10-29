"""AM-bench's 2-finger parallel slide gripper, shared joint names across the quad and hexa arms.

Bundled into both ``ambench_quad_arm`` and ``ambench_hexa_arm``'s own chain
(``manipulation_lfinger_link``/``manipulation_rfinger_link``, joints
``arm_lfinger_joint``/``arm_rfinger_joint``) -- confirmed via each bundle's
``conversion_report.json``. Ranges differ slightly between the two sources
(quad: 0 to 0.05 m; hexa: 0.001 to 0.045 m); this spec uses the quad's
numbers since both arms route through the same joint names here, and the
difference is small (5mm) -- note it rather than branch into two grippers
for a 5mm range difference.
"""

from .gripper_base import GripperSpec, register_gripper

AMBENCH_FINGER_GRIPPER = register_gripper(
    GripperSpec(
        name="ambench_finger_gripper",
        joint_names=("arm_lfinger_joint", "arm_rfinger_joint"),
        open_joint_positions=(0.0, 0.0),
        closed_joint_positions=(0.05, 0.05),
        usd_path=None,
        root_body_name=None,
        note=(
            "Bundled into ambench_quad_arm/ambench_hexa_arm's own chain; no USD yet "
            "either. Hexa's actual range is 0.001-0.045 m, slightly narrower than quad's 0-0.05 m."
        ),
    )
)
