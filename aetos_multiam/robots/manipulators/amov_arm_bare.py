"""The arm half of ``amov_drone_arm`` (see ``drones/amov_drone_arm.py``) with
its stock gripper stripped off, so the Inspire Hand can take over.

This is a "virtual" manipulator spec: there's no separate USD to mount
(``bundled_into_drone=True``, same situation as AM-bench's own arms) --
the arm is already part of ``amov_drone_arm``'s own USD. What this spec
adds on top of just "do nothing, it's bundled" (the AM-bench case) is
``deactivate_body_names``: ``link6`` (the stock gripper's mount body),
its sibling bodies ``gripper_left``/``gripper_right`` (the 2-finger
gripper itself, joints ``endleft``/``endright`` live on ``link6`` so they
go with it), the cosmetic ``ee_pin`` body, and ``joint6`` (lives on
``link5``, the DOF ``link6`` used to occupy) -- all deactivated by
``AerialManipulatorBase.spawn()`` before the Inspire Hand mounts onto
``link5`` via a Revolute joint that takes over that same wrist DOF
directly (see ``grippers/inspire_hand.py``'s ``INSPIRE_HAND_RIGHT`` for
the mount derivation -- it was derived against this exact USD already).

``arm_joint_names`` only lists joint1..5 -- joint6 no longer exists as
this arm's own joint once the Inspire Hand's mount joint replaces it.
"""

from .manipulator_base import ManipulatorSpec, register_manipulator

# Deliberately NOT imported from drones.amov_drone_arm -- manipulators/
# grippers must stay importable without Isaac Sim (see robots/__init__.py's
# docstring), and the drones package pulls in omni_drones transitively.
# Same literal path as drones/amov_drone_arm.py's USD_PATH; keep both in
# sync if that file ever moves.
from aetos_multiam.robots.drones.amov_drone_arm import USD_PATH as _AMOV_DRONE_ARM_USD_PATH  # noqa: E402

AMOV_ARM_BARE = register_manipulator(
    ManipulatorSpec(
        name="amov_arm_bare",
        usd_path=_AMOV_DRONE_ARM_USD_PATH,  # == amov_drone_arm's own USD, nothing separate to mount
        root_body_name="arm_base",
        eef_body_name="link5",
        arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5"),
        gripper_bundled=False,
        bundled_into_drone=True,
        default_gripper="inspire_hand_right",
        deactivate_body_names=("link5/joint6", "link6", "gripper_left", "gripper_right", "ee_pin"),
        note=(
            "Arm half of amov_drone_arm (OmniDrones' real asset) with link6 "
            "and its stock 2-finger gripper deactivated at spawn time so the "
            "Inspire Hand can mount on link5 instead."
        ),
    )
)
