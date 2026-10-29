"""airbot_play, 6-DOF arm with an integrated 2-finger gripper -- real asset.

Uses the self-contained USD (``my_airbot_play_v3_gripper.usd``, 4.3MB, all
meshes/materials inlined) rather than its 5KB sibling
``airbot_play_v3_gripper.usd``, which only references external relative
paths and would break once copied out of the DISCOVERSE checkout.

Joint/body names below are sourced from the MJCF at this repo's
``airbot_play/airbot_play.xml`` (same robot model): ``arm_base`` ->
``link1..link6`` via ``joint1..joint6``, with the gripper fingers
(``left``/``right`` bodies, ``endleft``/``endright`` joints) attached
directly to ``link6`` -- no separate gripper-mount body. The binary USD
itself couldn't be introspected in this sandbox (no working USD Python
bindings outside the Isaac Sim app), so **verify these names against the
actual USD stage the first time this runs in Isaac Sim**.
"""

from .manipulator_base import ManipulatorSpec, register_manipulator

USD_PATH = (
    "/home/benhua/DISCOVERSE/airbot_play_usd/my_airbot_play_v3_gripper/"
    "my_airbot_play_v3_gripper.usd"
)

AIRBOT_PLAY = register_manipulator(
    ManipulatorSpec(
        name="airbot_play",
        usd_path=USD_PATH,
        root_body_name="arm_base",
        eef_body_name="link6",
        arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        gripper_bundled=True,
        default_gripper="airbot_play_gripper",
        note=(
            "Real asset, used by the ButtonGrasp demo task. Body/joint names "
            "are sourced from the sibling MJCF, not a direct USD read -- "
            "double check against the live stage."
        ),
    )
)
