"""Dynamixel 2R arm -- real, rigid USD built via Isaac Sim's own MJCF importer.

Source: ``assets/mjcf/hummingbird_2r_three_finger_50pct/robot.xml`` (a
50%-scale combo with the three-finger gripper, mounted on an OmniDrones
Hummingbird). ``split_combo.py`` in that directory pulls the ``arm_carrier``
subtree out (dropping the gripper's own ``gripper_base`` body, keeping
``tool_adapter`` as an empty mount point) into a standalone
``dynamixel_2r_arm.xml``; ``import_arm_usd.py`` in the same directory runs
that through ``omni.importer.mjcf`` (plain rigid links + 2 hinges only, no
flexcomp/equality -- both already removed along with the gripper) and
cleans up the importer's own standalone-testing scaffolding (a world-anchor
fixed joint + a redundant empty ``worldBody`` ArticulationRootAPI holder,
same category of cleanup this project's other vendored assets needed).

Body chain: ``arm_carrier`` (root, no offset from the drone's own
``base_link``) -> ``base_2`` (motor housing, fixed-welded, no DOF) ->
``first_segment`` via hinge ``R1`` (±110 deg) -> ``second_segment`` via
hinge ``R2`` (±150 deg) -> ``tool_adapter`` (gripper mounts here, also
fixed-welded, no DOF of its own). Root/eef body names below carry an extra
``arm_carrier/`` segment because the importer always nests the MJCF's own
root body one level under whatever wrapper prim_path it's given (confirmed
empirically, not avoidable by renaming the wrapper) -- a relative two-segment
path works fine for ``mount_child``, just worth knowing if this ever looks
like a typo.

Joint drive gains (stiffness=50, damping=5.0, set directly on the imported
USD) come from the source MJCF's own ``<position class="dynamixel_2r"
kp="50" dampratio="1" .../>`` actuator -- kp maps directly to stiffness;
dampratio=1 (critical damping) has no simple closed form without
replicating MuJoCo's own per-joint-inertia calculation, so damping=5.0 is a
documented approximation, not derived -- tune at the controller level if
tracking looks off.
"""

import os.path as osp

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .manipulator_base import ManipulatorSpec, register_manipulator

MJCF_SOURCE_PATH = osp.join(
    ASSET_PATH, "mjcf", "hummingbird_2r_three_finger_50pct", "dynamixel_2r_arm.xml"
)
USD_PATH = osp.join(_ASSETS_USD_DIR, "dynamixel_2r", "dynamixel_2r_arm.usd")

DYNAMIXEL_2R = register_manipulator(
    ManipulatorSpec(
        name="dynamixel_2r",
        usd_path=USD_PATH,
        root_body_name="arm_carrier/arm_carrier",
        eef_body_name="arm_carrier/tool_adapter",
        arm_joint_names=("R1", "R2"),
        gripper_bundled=False,
        default_gripper="three_finger_gripper",
        # arm_carrier sits at the drone's own base_link origin with no
        # offset/rotation in the source MJCF (no pos/quat authored on that
        # body) -- identity mount.
        mount_translation=(0.0, 0.0, 0.0),
        mount_orientation=(1.0, 0.0, 0.0, 0.0),
        note=(
            f"Real USD imported from {MJCF_SOURCE_PATH} via omni.importer.mjcf "
            "(see import_arm_usd.py in the same directory). 50%-scale combo's "
            "own joint ranges/geometry, not necessarily the full-size arm's."
        ),
    )
)
