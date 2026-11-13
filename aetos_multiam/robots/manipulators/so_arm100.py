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
        # Z = -0.072839416 is robot.xml's own authored mount offset for the
        # ORIGINAL SO-ARM100's "Base" body -- not used as-is (see below).
        #
        # First attempt used the GLOBAL bounding-box minimum of the whole
        # drone body vs. the whole arm 'base' body to close an observed
        # 14.88mm gap (-0.0579581436631473) -- WRONG: the Iris frame's
        # global AABB-lowest point is out at a rotor arm, not under the
        # mount. Confirmed directly from the drone's own imported mesh
        # points (iris_frame_visual): within the arm's own XY footprint
        # (|x|<0.06, -0.09<y<0.05, centered on the mount), the drone body's
        # local lowest surface is z=1.44616 (baseline frame) --
        # 34.2mm *higher* than the global mesh minimum (z=1.41196) the
        # first fix used, because the body curves upward right above the
        # mount rather than staying flat. A close-up screenshot after the
        # first "fix" still showed a real visible gap, which is what caught
        # this.
        #
        # Redone properly: SO101's 'base' mesh-top sits
        # -0.0300811767670224 m below its own kinematic origin (measured at
        # the first fix's translation and cross-checked against
        # diagnose_so101_base_mesh.py's live origin/mesh readout -- this
        # offset is an asset-intrinsic constant, independent of
        # mount_translation). Solving so the mesh top lands exactly on the
        # drone's *local* surface under the mount:
        #   target_origin_z_rel_base_link = (1.44616 - 1.5) - (-0.0300811767670224)
        #                                  = -0.0538394153118134 + 0.0300811767670224
        #                                  = -0.023758238544791
        # Re-verified with scripts/measure_mount_gap.py (patched to sample
        # drone surface height within the arm's own footprint instead of
        # the whole-body AABB min) and a close-up render
        # (mount_junction_closeup.png) -- flush, no visible gap.
        #
        # Y = 0.02079: after the Z fix, user caught a sideways offset in the
        # live GUI -- same root cause as the Z issue, just the other axis.
        # mount_translation.y=0 put the arm ORIGIN exactly on the drone's Y
        # centerline (confirmed: both at world Y=-0.00001), but the SO101
        # 'base' MESH's own center sits at Y=-0.02080 relative to that
        # origin (scripts/measure_mount_y_offset.py), so the visible mesh
        # was still off-center by that amount even with a "centered" origin
        # -- the same asset-intrinsic origin-vs-mesh offset pattern as the Z
        # axis, not a coincidence. mount_translation is applied as a plain
        # offset in the parent (drone) frame, independent of
        # mount_orientation's rotation (confirmed from usd_compose.py's own
        # composition order), so this is a direct +0.02079 add, no further
        # transform needed.
        mount_translation=(0.0, 0.02079, -0.023758238544791),
        # Derived directly from robot.xml's own authored transforms (not the
        # README prose): "Base" (the original SO-ARM100's own root body) is
        # a child of "arm_carrier" (quat (0.70710678,0,0,-0.70710678) == -90
        # deg about Z, relative to iris_650), itself with pos (0,0,-0.0728...)
        # quat (0,1,0,0) == 180 deg about X relative to arm_carrier. Composed
        # (world_quat = parent_quat * local_quat, MJCF's own nesting
        # convention) via Gf.Quatd multiplication:
        #   (0.70710678,0,0,-0.70710678) * (0,1,0,0) = (0, 0.70710678,
        #   -0.70710678, 0)
        # Applied to the SO101 substitute (not the original SO-ARM100 this
        # number was derived for -- see module docstring); visually verified
        # via scripts/verify_iris650_soarm100.py: gripper ends up below and
        # forward of the drone body, not clipping through it. The rotation
        # itself needed no adjustment -- only the Z translation had a gap.
        mount_orientation=(0.0, 0.7071067811865476, -0.7071067811865476, 0.0),
        # Mounted flush against the airframe, the shoulder link's collider
        # overlaps the drone body's. That contact's friction held shoulder_pan
        # still against any drive torque (long misread as a PhysX drive defect).
        no_collide_with_drone=("shoulder",),
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
