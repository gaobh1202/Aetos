"""airbot_play's own 2-finger parallel gripper -- bundled into the arm USD.

Joint ranges sourced from the sibling MJCF (``airbot_play/airbot_play_dependencies.xml``):
``endleft`` (class ``finger2``) slides 0 -> 0.04, ``endright`` (class
``finger1``) slides -0.04 -> 0; both at 0 is open, at their extremes is
closed. No separate USD/spawn -- ``usd_path`` is ``None`` because the
fingers are already part of ``airbot_play``'s own USD.
"""

from .gripper_base import GripperSpec, register_gripper

AIRBOT_PLAY_GRIPPER = register_gripper(
    GripperSpec(
        name="airbot_play_gripper",
        joint_names=("endleft", "endright"),
        open_joint_positions=(0.0, 0.0),
        closed_joint_positions=(0.04, -0.04),
        usd_path=None,
        root_body_name=None,
        note="Bundled into the airbot_play manipulator USD; verify joint names against the live stage.",
    )
)
