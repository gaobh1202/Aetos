"""AmovDrone + its baked airbot_play arm+gripper -- OmniDrones' own combined asset.

Unlike ``amov_950`` (bare drone body, arm mounted at runtime via
``mount_child``), this points directly at
``/home/benhua/OmniDrones/amov_drone_arm_1.usd`` -- the literal file
OmniDrones' own ``task=HoverAmovDynaArmReplay`` checkpoint trained against.

Why this exists instead of just reusing ``amov_950`` + the ``airbot_gripper``/
``airbot_dexterous`` manipulator specs: those mount a SEPARATELY vendored
copy of the airbot_play arm (``assets/usd/airbot_play/``) whose ``arm_base``
body masses 0.777kg, vs. 0.166kg on this real training asset -- a mismatch
that also doesn't show up as "just" a mass error, since PhysX caches the
inertia tensor at parse time and never recomputes it from a bare
``set_body_masses()`` call. Confirmed in practice: an aetos_multiam shelf-
grasp reproduction script flew a real trained hover policy on the vendored
asset and it eventually flipped; swapping to this file (no other change)
made it match OmniDrones' own baseline run bit-for-bit. See
``docs/ARCHITECTURE.md`` for the full writeup.

Body/joint names on this USD (flat siblings under the drone's own prim, not
nested): ``arm_base`` -(joint1)-> ``link1`` -(joint2)-> ``link2`` -(joint3)->
``link3`` -(joint4)-> ``link4`` -(joint5)-> ``link5`` -(joint6)-> ``link6``,
with ``link6``'s own ``endleft``/``endright`` prismatic joints driving
sibling bodies ``gripper_left``/``gripper_right`` (the stock 2-finger
gripper), plus a cosmetic ``ee_pin`` body welded to ``link6`` via a Fixed
joint. ``amov950_airbot_bottom`` uses this whole chain as-is (manipulator/
gripper both ``None`` -- nothing to mount, the real gripper is already
part of this file). ``amov950_airbot_dexterous_bottom`` instead mounts the
Inspire Hand onto ``link5`` via the ``amov_arm_bare`` manipulator spec,
which deactivates ``link6``/``gripper_left``/``gripper_right``/``ee_pin``
and ``joint6`` first -- see ``manipulators/amov_arm_bare.py``.
"""

import os.path as osp

from omni_drones.robots.drone.amov_drone import AmovDrone as _OmniDronesAmovDrone
from aetos_multiam.assets import ASSET_PATH as _AETOS_ASSET_PATH, USD_PATH as _AETOS_USD_PATH

from .drone_base import DroneSpec, register_drone

USD_PATH = osp.join(_AETOS_USD_PATH, "amov_drone_arm", "amov_drone_arm_1.usd")  # copied from the OmniDrones fork
PARAM_PATH = osp.join(_AETOS_ASSET_PATH, "usd", "amov_drone_arm_1.yaml")  # copied from the OmniDrones fork


class AmovDroneArm(_OmniDronesAmovDrone):
    """950mm AmovDrone with the airbot_play arm+gripper pre-baked in -- the
    real asset, not a runtime-mounted stand-in."""

    usd_path: str = USD_PATH
    param_path: str = PARAM_PATH
    hover_target_usd_path: str = USD_PATH


AMOV_DRONE_ARM = register_drone(
    DroneSpec(
        name="amov_drone_arm",
        drone_class=AmovDroneArm,
        usd_path=USD_PATH,
        param_path=PARAM_PATH,
        mount_body_name="link5",
        class_size_mm=950,
        eef_body_name="ee_pin",  # welded to link6 via the "ee_mount" fixed joint
        bundled_arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        # The baked-in 2-finger gripper's own DOFs (live dof_names:
        # rotor_joint_1..4, joint1..6, endleft, endright).
        bundled_gripper_joint_names=("endleft", "endright"),
        note=(
            "Real asset: OmniDrones' own amov_drone_arm_1.usd (the file "
            "task=HoverAmovDynaArmReplay actually trained against), "
            "arm+gripper pre-baked in."
        ),
    )
)
