"""SO-ARM100/SO101's gripper -- its 6th joint, bundled into the arm's own chain.

No separate mount body -- the moving jaw (``jaw``) is a normal sibling body
driven by the ``gripper`` joint in ``so_arm100``'s own kinematic chain (see
``manipulators/so_arm100.py``, which actually spawns Lightwheel's SO101
substitute USD), so ``usd_path``/``root_body_name`` stay ``None`` just like
``airbot_play_gripper``.

Open/closed values switched from the original SO-ARM100 MJCF source's
numbers (0.8 / -0.174 rad on a joint named ``Jaw``) to the SO101 substitute
USD's own joint (named ``gripper``, not ``Jaw``) and its authored limits --
``leisaac``'s own config lists this joint's USD range as -10..100 degrees
(``SO101_FOLLOWER_USD_JOINT_LIMLITS["gripper"]``), i.e. roughly
-0.1745..1.745 rad. Picked closed=0.0 rad (near the USD lower bound) and
open=1.5 rad (near the upper bound, leaving a small margin) -- not verified
against a live grasp test yet, just kept inside the authored limits.
"""

from .gripper_base import GripperSpec, register_gripper

SO_ARM100_GRIPPER = register_gripper(
    GripperSpec(
        name="so_arm100_gripper",
        joint_names=("gripper",),
        open_joint_positions=(1.5,),
        closed_joint_positions=(0.0,),
        usd_path=None,
        root_body_name=None,
        note=(
            "Bundled into the so_arm100 manipulator's own chain (which "
            "actually spawns the SO101 substitute USD, see so_arm100.py). "
            "Joint renamed from the SO-ARM100 source's 'Jaw' to the SO101 "
            "USD's actual 'gripper' joint name; open/closed values picked "
            "to stay inside its authored -10..100 deg USD limit, not "
            "copied from the SO-ARM100 source's different numbers."
        ),
    )
)
