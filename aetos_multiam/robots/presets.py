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
    # Mount: bottom. Not stated as explicitly as the others below (no source
    # bundle doc to cite) -- inferred from this repo's own project memory,
    # where the amov_drone_arm_1 combined asset's arm-base joint force has
    # Fz < 0 (the arm's weight pulling down on the joint), consistent with
    # the arm hanging below the body rather than sitting on top.
    "amov950_airbot_bottom": AerialManipulatorCfg(
        drone_type="amov_950",
        manipulator_type="airbot_play",
        gripper_type=None,  # bundled into airbot_play's own USD
    ),
    # Mount: bottom. Source: hummingbird_2r_three_finger_50pct/README.md --
    # "下挂机械臂" (arm hangs below).
    "hummingbird340_dynamixel2r_bottom": AerialManipulatorCfg(
        drone_type="omnidrones_340",
        manipulator_type="dynamixel_2r",
        gripper_type="three_finger_softgripper",
    ),
    # Mount: bottom. Source: iris_650_so_arm100_final/README.md -- "刚性
    # 安装在中央机身下方" (rigidly mounted below the central body).
    "iris650_soarm100_bottom": AerialManipulatorCfg(
        drone_type="iris_650",
        manipulator_type="so_arm100",
        gripper_type=None,  # bundled into so_arm100's own chain (the "Jaw" joint)
    ),
    # Mount: top. Source: ambench_quad_600_top_arm_final/CONVERSION_NOTES.md
    # -- arm explicitly flipped 180 deg and re-mounted top-side-up on request.
    "uaquad600_ambencharm_top": AerialManipulatorCfg(
        drone_type="ambench_600",
        manipulator_type="ambench_quad_arm",
        gripper_type=None,  # bundled into ambench_quad_arm's own chain
    ),
    # Mount: top. Source: ambench_fa_hexa_750_final/CONVERSION_NOTES.md --
    # "六旋翼保持原上置装配" (hexacopter kept its original top-mounted layout).
    "fahexa750_ambencharm_top": AerialManipulatorCfg(
        drone_type="ambench_750",
        manipulator_type="ambench_hexa_arm",
        gripper_type=None,  # bundled into ambench_hexa_arm's own chain
    ),
}


def get_preset(name: str) -> AerialManipulatorCfg:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}") from None
