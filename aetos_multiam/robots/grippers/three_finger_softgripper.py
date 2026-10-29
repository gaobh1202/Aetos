"""Three-finger soft (Fin Ray) gripper -- placeholder, source asset not Isaac-Sim-usable.

Two MJCF sources now live under ``assets/mjcf/``: the original full-size
gripper-only ``three_finger_gripper_final`` reference (not copied into this
repo) and the 50%-scale combo mounted on a Hummingbird + dynamixel 2R in
``hummingbird_2r_three_finger_50pct/robot.xml`` -- confirmed (grep'd) to
still use the same ``flexcomp`` soft-body fingers as the original. PhysX /
Isaac Sim has no equivalent to MuJoCo flex soft bodies, so neither can be
imported as-is -- this needs a rigid-body approximation (e.g. a single rigid
finger segment per joint, or a few pin-jointed rigid links) authored as a
new USD before this spec is real. Joint layout below (one driven hinge +
two mimic joints, ``finger_2_synchronized``/``finger_3_synchronized``
equality constraints) is accurate and taken directly from the source MJCF's
``<equality>`` block, so the *kinematics* are already right; only the
compliant-finger *geometry* needs the rigid approximation. Ranges below are
the 50%-scale combo's (same numeric range as the original -- MuJoCo
``<equality>`` mimics don't rescale joint limits, only geometry/mass).
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .gripper_base import GripperSpec, register_gripper

MJCF_SOURCE_PATH = osp.join(
    ASSET_PATH, "mjcf", "hummingbird_2r_three_finger_50pct", "robot.xml"
)
USD_PATH = osp.join(_ASSETS_USD_DIR, "three_finger_softgripper.usd")  # does not exist yet
_FULL_CLOSE_RAD = 0.6981317008  # 40 deg, from the source MJCF's joint range

THREE_FINGER_SOFTGRIPPER = register_gripper(
    GripperSpec(
        name="three_finger_softgripper",
        joint_names=("finger_1_hinge", "finger_2_hinge", "finger_3_hinge"),
        open_joint_positions=(0.0, 0.0, 0.0),
        closed_joint_positions=(_FULL_CLOSE_RAD, _FULL_CLOSE_RAD, _FULL_CLOSE_RAD),
        usd_path=USD_PATH,
        root_body_name="gripper_base",
        note=(
            f"Real MJCF source at {MJCF_SOURCE_PATH}; still uses flexcomp "
            "soft fingers with no PhysX equivalent -- needs a "
            "rigid-approximation USD authored before use."
        ),
    )
)
