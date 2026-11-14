"""Do the kinematic models in arm_kinematics.py describe the arms the simulator flies?

For each preset: the model's end-effector pose in the drone base frame against the simulator's
link poses (recorded by debug/dump_arm_link_poses.py), then inverse kinematics on those poses.

    conda activate armplan;  python scripts/arm_planning/check_arm_kinematics.py
"""
import os.path as osp

import numpy as np
from scipy.spatial.transform import Rotation as R

from arm_kinematics import ARMS, MJCF_DIR, ArmKinematics

DATA = osp.join(osp.dirname(MJCF_DIR), "..", "data", "arm_kinematics")


def T(pose):  # (x, y, z, qw, qx, qy, qz)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R.from_quat(np.roll(pose[3:], -1)).as_matrix(), pose[:3]
    return m


ok = True
for preset in ARMS:
    d = np.load(osp.join(DATA, f"{preset}.npz"))
    kin = ArmKinematics(preset)
    links, dofs = list(d["link_names"]), list(d["dof_names"])
    cols = [dofs.index(j) for j in kin.joint_names]
    ee = links.index(kin.m.body(kin.ee).name)
    e_pos, e_rot, solved, e_ik = [], [], 0, []
    position_only = kin.n < 5
    for q_all, poses in zip(d["q"], d["link_poses"]):
        q = q_all[cols]
        sim = np.linalg.inv(T(poses[links.index("base_link")])) @ T(poses[ee])
        fk = kin.fk(q)
        e_pos.append(np.linalg.norm(fk[:3, 3] - sim[:3, 3]))
        e_rot.append(np.linalg.norm(R.from_matrix(fk[:3, :3].T @ sim[:3, :3]).as_rotvec()))
        sols = kin.ik(sim, q0=q + 0.2, position_only=position_only, pos_tol=1e-3, rot_tol=1e-2)
        solved += bool(sols)
        if sols:  # the simulator's own posture is one of the solutions; how far is the nearest one found
            e_ik.append(min(np.abs(s - q).max() for s in sols))
    n = len(d["q"])
    good = max(e_pos) < 1e-3 and max(e_rot) < 1e-2 and solved == n
    ok &= good
    print(f"{preset}: forward vs simulator, worst of {n} postures {max(e_pos) * 1000:.3f} mm / {np.degrees(max(e_rot)):.3f} deg; "
          f"inverse ({'position only' if position_only else 'full pose'}) solved {solved}/{n}"
          f"{f', nearest solution within {np.degrees(max(e_ik)):.2f} deg of the simulator posture' if e_ik and not position_only else ''}"
          f"  {'OK' if good else 'FAIL'}")
print("RESULT:", "OK" if ok else "FAIL")
