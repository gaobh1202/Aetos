"""Leader motor reading -> sim joint target, for the SO101 leader -> so_arm100 pair.

Same linear remap as leisaac's ``convert_action_from_so101_leader``
(devices/action_process.py): rescale from the leader's own normalized device
range into the follower USD's authored joint range (degrees), then
degrees -> radians. Leader and follower are the same hardware family
(LeRobot SO-ARM100/SO101), so this is a direct per-joint mirror -- no IK.
"""

import math
from typing import Dict, Tuple

JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# USD joint limits in degrees -- verified directly against this project's own
# aetos_multiam/assets/meshes/usd/so101_follower/so101_follower.usd (read via
# UsdPhysics.Joint's physics:lowerLimit/upperLimit attributes), not just
# copied from leisaac's docs -- confirmed to match exactly, since it's the
# same asset file (see so_arm100.py's module docstring: both projects use
# Lightwheel's so101_follower.usd release asset).
SO101_USD_JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}


def convert_leader_state_to_radians(
    joint_state: Dict[str, float],
    motor_limits: Dict[str, Tuple[float, float]],
    usd_joint_limits_deg: Dict[str, Tuple[float, float]] = SO101_USD_JOINT_LIMITS_DEG,
) -> Dict[str, float]:
    """``joint_state``: leader's raw ``Present_Position`` reading (normalized
    device units, e.g. from ``SOArmLeader.read_joint_state()``).

    Returns ``{joint_name: radians}``, one entry per ``JOINT_ORDER`` name.
    The caller is responsible for how it's applied to the sim (this
    project's own amteleop_soarm_leader.py example teleports joint
    positions directly rather than setting PD targets -- see its module
    docstring for why).
    """
    out = {}
    for name in JOINT_ORDER:
        motor_lo, motor_hi = motor_limits[name]
        joint_lo, joint_hi = usd_joint_limits_deg[name]
        motor_range = motor_hi - motor_lo
        joint_range = joint_hi - joint_lo
        frac = (joint_state[name] - motor_lo) / motor_range
        degrees = frac * joint_range + joint_lo
        out[name] = math.radians(degrees)
    return out
