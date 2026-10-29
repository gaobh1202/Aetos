"""lerobot SO-ARM100 -- real USD now exists, but it's actually an **SO101**.

The original MJCF source at ``assets/mjcf/iris_650_so_arm100_final/robot.xml``
is a genuine full-size SO-ARM100 (joint names ``Rotation``/``Pitch``/
``Elbow``/``Wrist_Pitch``/``Wrist_Roll``/``Jaw``), mounted on a 650mm-scaled
OmniDrones Iris (see ``drones/iris_650.py``) -- but no one has converted
*that specific* MJCF to USD. What's registered here instead is a
**substitute**: Lightwheel's own ``so101_follower.usd``
(https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd,
~22MB, downloaded with the user's explicit go-ahead), built for their
``leisaac`` Isaac Lab project. SO101 is a later hardware revision of the
same LeRobot/TheRobotStudio arm family -- mechanically close (same 5-DOF-arm
+ 1-DOF-gripper layout) but not identical to SO-ARM100 (servo/joint specifics
differ); this is a deliberate, documented substitution, not the real
SO-ARM100 geometry the rest of this project's numbers (mount offset, home
pose) were derived against.

Verified by actually loading it in Isaac Sim (``add_reference_to_stage`` +
``Articulation.initialize()``): a clean, already-correctly-authored PhysX
articulation (flat sibling bodies, no nested-RigidBody or LocalRot0/1 issues
of the kind hand-authoring ran into with the fin-ray finger) --
``dof_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
'wrist_roll', 'gripper']`` (LeRobot's own naming, NOT SO-ARM100's
``Rotation``/``Pitch``/... convention -- do not mix the two up when reading
this project's other notes about SO-ARM100's joint names). Body tree:
``base`` (root) -> ``shoulder`` -> ``upper_arm`` -> ``lower_arm`` -> ``wrist``
-> ``gripper`` -> ``jaw`` (the moving jaw, driven by the ``gripper`` joint).

Mount geometry below is still the SO-ARM100 bundle's own numbers (not
re-derived for SO101's slightly different link lengths) -- treat as
approximate until checked against the live SO101 mesh once actually mounted.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .manipulator_base import ManipulatorSpec, register_manipulator

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "iris_650_so_arm100_final", "robot.xml")
USD_PATH = osp.join(_ASSETS_USD_DIR, "so101_follower", "so101_follower.usd")

HOME_ANGLES_RAD = (0.0, -1.57, 1.57, 1.57, -1.57)  # SO-ARM100 source's own home pose, in its own joint order

SO_ARM100 = register_manipulator(
    ManipulatorSpec(
        name="so_arm100",
        usd_path=USD_PATH,
        root_body_name="base",
        eef_body_name="jaw",
        arm_joint_names=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        gripper_bundled=True,
        default_gripper="so_arm100_gripper",
        mount_translation=(0.0, 0.0, -0.072839416),
        # mount_orientation left unset: the source bundle composes a 180 deg
        # flip about the arm's own X with a further -90 deg yaw about the
        # drone's body Z (see docstring). Deriving one quaternion for that is
        # straightforward but unverified against the live asset -- compute
        # and check it against the actual mounted pose once this is wired up,
        # rather than trusting an un-tested derivation here.
        note=(
            f"Real SO-ARM100 MJCF source at {MJCF_SOURCE_PATH} (full "
            "original size, so_arm100_scale=1.0) -- but the spawnable USD "
            "is a SO101 substitute (see module docstring), not a conversion "
            "of that MJCF. Servo params from the SO-ARM100 MJCF source "
            "(not re-derived for SO101): kp=50, dampratio=1, armature=0.1, "
            "frictionloss=0.1, torque limit +/-3.5 N*m."
        ),
    )
)
