"""Three-finger Fin Ray gripper -- real, rigid-body asset.

MuJoCo's own version of this gripper uses ``flexcomp`` soft-body fingers,
which PhysX/Isaac Sim has no equivalent for (see
``docs/ARCHITECTURE.md``'s "Three-finger gripper" history for the full
investigation, including a separate attempt at an actual PhysX Deformable
Body version of this exact gripper that got stuck on a legacy-API
attachment bug -- abandoned per explicit user direction in favor of this
rigid approximation). This spec is the rigid version: same outer geometry
(mount bracket + finger "blade" -- the flexcomp's own surface mesh, just not
simulated as soft), one rigid body per finger, each driven by a
``PhysicsRevoluteJoint`` using the original hinge's own pivot/axis/limits.
No deformable schema, no attachment, no GPU-dynamics requirement.

Built by ``aetos_multiam/assets/mjcf/hummingbird_2r_three_finger_50pct/convert_gripper_rigid.py``
from that bundle's own 50%-scale ``three_finger_gripper_50pct.xml`` (itself
split out of ``robot.xml`` by ``split_combo.py`` in the same directory --
same gripper design as the original full-size ``three_finger_gripper_final``
reference, just at this bundle's own scale). Joint layout (one driven hinge
per finger, no MuJoCo ``<equality>`` mimic constraints carried over --
each of the 3 joints here is independently driven, matching the pattern
already used for the standalone full-scale rigid gripper prototype) and
ranges are read directly from the source MJCF.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .gripper_base import GripperSpec, register_gripper

MJCF_SOURCE_PATH = osp.join(
    ASSET_PATH, "mjcf", "hummingbird_2r_three_finger_50pct", "three_finger_gripper_50pct.xml"
)
USD_PATH = osp.join(_ASSETS_USD_DIR, "three_finger_gripper_50pct_rigid", "gripper_rigid.usda")
_FULL_CLOSE_RAD = 0.6981317008  # 40 deg, from the source MJCF's joint range

THREE_FINGER_GRIPPER = register_gripper(
    GripperSpec(
        name="three_finger_gripper",
        joint_names=("finger_1", "finger_2", "finger_3"),
        open_joint_positions=(0.0, 0.0, 0.0),
        closed_joint_positions=(_FULL_CLOSE_RAD, _FULL_CLOSE_RAD, _FULL_CLOSE_RAD),
        usd_path=USD_PATH,
        root_body_name="Housing",
        # tool_adapter -> gripper_base offset in the source combo MJCF
        # (gripper_base itself carries no rotation relative to tool_adapter).
        mount_translation=(0.0, 0.0, 0.008),
        mount_orientation=(1.0, 0.0, 0.0, 0.0),
        note=(
            f"Real MJCF source at {MJCF_SOURCE_PATH}. Rigid approximation -- "
            "each finger is one rigid body (mount bracket + blade combined), "
            "convexHull collision; not a validated grasp-force model, just "
            "matching structure/shape (see note above re: the abandoned "
            "deformable attempt)."
        ),
    )
)
