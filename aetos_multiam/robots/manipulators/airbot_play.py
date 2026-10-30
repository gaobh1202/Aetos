"""airbot_play -- 6-DOF arm, two variants sharing the same vendored USD family.

``airbot_gripper``: the full chain (``arm_base -> link1..link6``, joint1..6)
with the original 2-finger gripper (``left``/``right`` bodies, ``endleft``/
``endright`` joints) already part of link6 -- gripper_bundled=True, nothing
to mount.

``airbot_dexterous``: the SAME arm with the whole of link6 removed (not just
its fingers) -- the Inspire Hand takes over link5's own wrist joint
(``joint6``) directly via a Revolute mount, "able to rotate" the same way
link6 used to -- see ``grippers/inspire_hand.py``'s ``INSPIRE_HAND_RIGHT``
for the mount derivation. ``arm_joint_names`` only lists joint1..5 since
joint6 no longer exists as this arm's own joint (it's now the gripper's
mount joint).

Both USDs are vendored locally (``aetos_multiam/assets/usd/airbot_play/``),
not the original DISCOVERSE source checkout -- see docs/ARCHITECTURE.md for
the full history of fixes applied to them (the ``endleft``/``endright``
finger-orientation fix, the link6/joint6 removal, and the removal of a
``root_joint``->``world`` world-anchor chain both copies originally had for
standalone testing).

``MOUNT_TRANSLATION``/``MOUNT_ORIENTATION`` (drone ``base_link`` -> this
arm's ``arm_base``) are read off ``/home/benhua/OmniDrones/amov_drone_arm_1.usd``'s
own ``base_link -> arm_mount_link -> arm_base`` joint chain (a known-good,
hand-built amov_950+airbot combo from the sibling OmniDrones repo) --
180 degrees about local X, since the arm hangs upside-down/below the drone
body. Verified by un-rotating that reference's own link6 position and
matching it against this project's vendored copy's link6-relative-to-
arm_base position to 4 decimal places (confirms same underlying arm asset).
"""

import os.path as osp

from aetos_multiam.assets import USD_PATH as _ASSETS_USD_DIR
from .manipulator_base import ManipulatorSpec, register_manipulator

_AIRBOT_USD_DIR = osp.join(_ASSETS_USD_DIR, "airbot_play")
GRIPPER_USD_PATH = osp.join(_AIRBOT_USD_DIR, "airbot_play_gripper.usd")
BARE_USD_PATH = osp.join(_AIRBOT_USD_DIR, "airbot_play_bare.usd")

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

MOUNT_TRANSLATION = (0.005561, -0.002454, -0.125)
MOUNT_ORIENTATION = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z): 180 deg about local X

AIRBOT_GRIPPER = register_manipulator(
    ManipulatorSpec(
        name="airbot_gripper",
        usd_path=GRIPPER_USD_PATH,
        root_body_name="arm_base",
        eef_body_name="link6",
        arm_joint_names=ARM_JOINT_NAMES,
        gripper_bundled=True,
        default_gripper="airbot_play_gripper",
        mount_translation=MOUNT_TRANSLATION,
        mount_orientation=MOUNT_ORIENTATION,
        note=(
            "Full airbot_play chain (arm_base+link1..6) with its own stock "
            "2-finger gripper (left/right bodies off link6) bundled in."
        ),
    )
)

AIRBOT_DEXTEROUS = register_manipulator(
    ManipulatorSpec(
        name="airbot_dexterous",
        usd_path=BARE_USD_PATH,
        root_body_name="arm_base",
        eef_body_name="link5",
        arm_joint_names=ARM_JOINT_NAMES[:5],
        gripper_bundled=False,
        default_gripper="inspire_hand_right",
        mount_translation=MOUNT_TRANSLATION,
        mount_orientation=MOUNT_ORIENTATION,
        note=(
            "Same arm with link6 (and its now-dangling joint6) entirely "
            "removed -- the Inspire Hand (inspire_hand_right) mounts onto "
            "link5 via a Revolute joint taking over link6/joint6's own "
            "wrist DOF, not a Fixed weld onto a vestigial tool frame."
        ),
    )
)
