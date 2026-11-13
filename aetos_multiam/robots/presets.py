"""Named aerial-manipulator presets -- one per pre-built combo asset bundle.

Naming convention: ``<drone size+family><manipulator type>_<mount position>``
-- e.g. ``amov950_airbot_bottom`` = 950mm Amov drone, airbot_play arm,
mounted below the body. Mount position (``top``/``bottom``) is taken from
each source bundle's own documentation (see each preset's comment below),
not guessed. Each preset is just an ``AerialManipulatorCfg``; nothing here
is special beyond grouping the five combinations this project has source
assets for into discoverable names, mirroring robosuite's convention of a
few canonical ready-to-use robot configs. See ``docs/ARCHITECTURE.md``'s
asset table for which of these can actually be spawned today (only
``amov950_airbot_bottom``) vs. which are registered-but-pending-USD-conversion.

These were registered under different, source-bundle-derived names in an
earlier pass (``amov_airbot``, ``hummingbird_dynamixel_three_finger``,
``iris650_so_arm100``, ``uaquad600_top_arm``, ``fahexa750_arm``) -- renamed
here for a consistent scheme; update any external references.
"""

from typing import Dict

from .robot_base import AerialManipulatorCfg

PRESETS: Dict[str, AerialManipulatorCfg] = {
    # Mount: bottom. Drone+arm+stock-gripper all come from one file --
    # OmniDrones' own amov_drone_arm_1.usd (``amov_drone_arm`` drone spec),
    # not amov_950 + the separately-vendored airbot_gripper manipulator.
    # That vendored copy's arm_base body masses 4.7x the real training
    # asset's (0.777kg vs 0.166kg) and PhysX doesn't recompute its cached
    # inertia tensor from a bare mass fix -- confirmed to actually matter:
    # a trained hover policy flown on the vendored asset eventually flipped,
    # flown on this real asset it didn't. See drones/amov_drone_arm.py.
    "amov950_airbot_bottom": AerialManipulatorCfg(
        drone_type="amov_drone_arm",
        manipulator_type=None,  # arm+gripper already baked into amov_drone_arm's USD
        gripper_type=None,
        # 3 kg of real arm on a 9.85 kg drone: its reaction is genuine physics,
        # monotonic in the residual, and needs to come down two decades.
        # 1.0 -> 269 deg of yaw, 0.1 -> 98, 0.02 -> 18.5, 0.001 -> 4.5.
        arm_dynamics_residual=0.001,
    ),
    # Same real asset, but with the stock 2-finger gripper deactivated and
    # the Inspire Hand mounted on link5 instead -- see
    # manipulators/amov_arm_bare.py and grippers/inspire_hand.py.
    "amov950_airbot_dexterous_bottom": AerialManipulatorCfg(
        drone_type="amov_drone_arm",
        manipulator_type="amov_arm_bare",
        gripper_type=None,  # default_gripper="inspire_hand_right" on the manipulator spec
        # Same drone+arm as the gripper variant, so the same residual -- but
        # here it is not merely better, it is REQUIRED: the Inspire hand's 12
        # finger drives diverge at 0.1 and at 1.0 (joint positions reaching
        # 1e4 then 1e12 rad, "Invalid PhysX transform", NaN articulation),
        # and are stable at 0.001. Do not raise it without re-testing.
        arm_dynamics_residual=0.001,
        # ...and 0.001 is only enough to hold the hand STILL. Commanding the
        # fingers through their PD drives diverges within the first motion no
        # matter the residual: the asset authors them at kp=0.05..0.07,
        # kd=1e-4, alongside arm joints at kp=1e7 -- a 1e8 spread inside one
        # articulation. Scaling them down with the residual was tried and is
        # worse (the drone ended 19 m up at 154 deg of tilt). Teleport works:
        # the full scripted mission runs with 0.46 deg of body tilt. The
        # planned Apple Vision Pro retargeter has to use this path too.
        arm_write="teleport",
    ),
    # Mount: bottom. Source: hummingbird_2r_three_finger_50pct/README.md --
    # "下挂机械臂" (arm hangs below).
    "hummingbird340_2dof_gripper": AerialManipulatorCfg(
        drone_type="omnidrones_340",
        manipulator_type="dynamixel_2r",
        gripper_type=None,  # default_gripper="three_finger_gripper" on the manipulator spec
        # Barely needs it: fully coupled this arm only costs 8.3 deg of roll.
        # Linear in the residual, 0.1 -> 0.72 deg.
        arm_dynamics_residual=0.1,
    ),
    # Mount: bottom. Source: iris_650_so_arm100_final/README.md -- "刚性
    # 安装在中央机身下方" (rigidly mounted below the central body).
    "iris650_soarm100": AerialManipulatorCfg(
        drone_type="iris_650",
        manipulator_type="so_arm100",
        gripper_type=None,  # bundled into so_arm100's own chain (the "Jaw" joint)
        # NOT monotonic for this one -- 0.1 is a measured optimum and 0.02 is
        # already 8x worse. shoulder_pan was locked for teleop because its drive
        # looked dead; the cause was the shoulder collider overlapping the
        # airframe (fixed 2026-11-12 by so_arm100's no_collide_with_drone). The
        # lock is kept until teleop is re-measured without it.
        # docs/ARM_DISTURBANCE.md.
        arm_dynamics_residual=0.1,
        arm_locked_joints=("shoulder_pan",),
    ),
    # Mount: top. Source: ambench_quad_600_top_arm_final/CONVERSION_NOTES.md
    # -- arm explicitly flipped 180 deg and re-mounted top-side-up on request.
    "uaquad600_gripper": AerialManipulatorCfg(
        drone_type="ambench_600",
        manipulator_type="ambench_quad_arm",
        gripper_type=None,  # bundled into ambench_quad_arm's own chain
        # Verified to hover (1.96 deg tilt) and fly the keyboard sequence at
        # this value; the residual itself was not swept for this preset.
        arm_dynamics_residual=0.1,
        # quad_scorpion.usd mounts its arm UNDER the airframe (measured: every
        # arm body 0.05-0.29 m below base_link). This project's own MJCF bundle
        # for the same robot, ambench_quad_600_top_arm_final, mounts it on top
        # -- "arm explicitly flipped 180 deg and re-mounted top-side-up on
        # request", per its CONVERSION_NOTES. Same transform as that bundle's
        # own robot.xml authors on arm_base_link: pos (.1, 0, .003), quat
        # (0, 1, 0, 0) = 180 deg about X, against the USD's (0.1, 0, -0.053)
        # and identity.
        joint_pose_overrides={
            "joints/arm_base_fixed": ((0.1, 0.0, 0.003), (0.0, 1.0, 0.0, 0.0)),
        },
    ),
    # Mount: top. Source: ambench_fa_hexa_750_final/CONVERSION_NOTES.md --
    # "六旋翼保持原上置装配" (hexacopter kept its original top-mounted layout).
    "fahexa750_gripper": AerialManipulatorCfg(
        drone_type="ambench_750",
        manipulator_type="ambench_hexa_arm",
        gripper_type=None,  # bundled into ambench_hexa_arm's own chain
        # Verified to hover (0.29 deg tilt) and fly the keyboard sequence at
        # this value; the residual itself was not swept for this preset.
        # Fixed-cant fully-actuated hexa, but flown as an underactuated one --
        # exact 4-DOF mixer, unused lateral authority. See
        # docs/TELEOP_CONTROL_MODES.md.
        arm_dynamics_residual=0.1,
        # hexa_scorpion.usd authors enabledSelfCollisions=False (the other five
        # presets all author True), so its arm sweeps straight through the
        # airframe. AM-Bench's own ArticulationCfg for this robot asks for
        # True -- ambench/robots/fa_hexa.py.
        enable_self_collisions=True,
    ),
}


def get_preset(name: str) -> AerialManipulatorCfg:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}") from None
