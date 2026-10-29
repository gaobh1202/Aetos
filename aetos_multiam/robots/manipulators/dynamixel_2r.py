"""Dynamixel 2R arm -- real MJCF source now exists, USD conversion still pending.

Joint/body names and ranges below are read directly from
``assets/mjcf/hummingbird_2r_three_finger_50pct/robot.xml`` (a 50%-scale
combo with the three-finger gripper, mounted on an OmniDrones Hummingbird):
mount body ``arm_carrier`` -> ``base_2`` (motor housing, unjointed) ->
``first_segment`` via hinge ``R1`` (range ±1.9199 rad) -> ``second_segment``
via hinge ``R2`` (range ±2.6180 rad) -> ``tool_adapter`` (gripper mounts
here). These are the *50%-scaled* joint limits/geometry; the full-size
arm's own numbers aren't in that bundle (it only ships the scaled combo).

Still no USD -- MJCF uses `omni.isaac.core`-incompatible geometry (meshes
are `.stl`, fine for conversion, but there's no Isaac Sim import step run
yet). ``usd_path`` stays a placeholder.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .manipulator_base import ManipulatorSpec, register_manipulator

MJCF_SOURCE_PATH = osp.join(
    ASSET_PATH, "mjcf", "hummingbird_2r_three_finger_50pct", "robot.xml"
)
USD_PATH = osp.join(_ASSETS_USD_DIR, "dynamixel_2r.usd")  # does not exist yet

DYNAMIXEL_2R = register_manipulator(
    ManipulatorSpec(
        name="dynamixel_2r",
        usd_path=USD_PATH,
        root_body_name="base_2",
        eef_body_name="tool_adapter",
        arm_joint_names=("R1", "R2"),
        gripper_bundled=False,
        default_gripper="three_finger_softgripper",
        note=(
            f"Real MJCF source at {MJCF_SOURCE_PATH} (50%-scale combo on "
            "Hummingbird); no USD yet. Joint ranges here are the scaled "
            "combo's numbers, not necessarily the full-size arm's. The "
            "source MJCF has an 'arm_carrier' bracket body between the "
            "drone and this arm's own 'base_2' root -- whether that bracket "
            "belongs to the drone side or the arm side needs confirming "
            "before wiring up `mount_child`'s translation/orientation."
        ),
    )
)
