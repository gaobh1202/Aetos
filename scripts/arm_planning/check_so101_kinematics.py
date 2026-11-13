"""Does LeRobot's SO101 URDF describe the arm the simulator flies?

Compares the grasp point recorded in the demos (PhysX link poses) with
URDF forward kinematics through the drone pose and the mount, then round-trips
LeRobot's placo IK. Run in the ``armplan`` env:  python scripts/arm_planning/check_so101_kinematics.py
"""
import numpy as np

from lerobot_kinematics import RobotKinematics
from so101_common import ARM_JOINTS, Q_URDF_MINUS_SIM, T, T_DRONE_ARMBASE, T_GRIPPER_GRASP, URDF, load_demo

kin = RobotKinematics(URDF, "gripper_link", ARM_JOINTS)
worst = 0.0
for ep in range(10):
    d = load_demo(ep)
    for a in ("a0", "a1"):
        for k in range(50, len(d["t"]), 50):  # step 0 holds a stale grasp point
            fk = kin.forward_kinematics(np.degrees(d[f"{a}_arm_q"][k, :5] + Q_URDF_MINUS_SIM))
            p = (T(d[f"{a}_drone_pos"][k], d[f"{a}_drone_quat"][k]) @ T_DRONE_ARMBASE @ fk @ T_GRIPPER_GRASP)[:3, 3]
            worst = max(worst, float(np.linalg.norm(p - d[f"{a}_grasp_point"][k])))
print(f"FK vs simulator grasp point, worst over 10 demos: {worst * 1000:.2f} mm")
assert worst < 0.003, "URDF and simulated arm disagree"

# IK round trip: position-only targets reached from the demo pose.
rng = np.random.default_rng(0)
q0 = np.degrees(load_demo(0)["a0_arm_q"][100, :5] + Q_URDF_MINUS_SIM)
errs = []
for _ in range(200):
    target = kin.forward_kinematics(q0 + rng.uniform(-25, 25, 5)).copy()
    q = q0
    for _ in range(20):  # placo takes one QP step per call
        q = kin.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=0.0)
    errs.append(np.linalg.norm(kin.forward_kinematics(q)[:3, 3] - target[:3, 3]))
print(f"placo IK round trip (position only, 200 targets): mean {np.mean(errs) * 1000:.2f} mm, worst {np.max(errs) * 1000:.2f} mm")
assert np.max(errs) < 0.002
