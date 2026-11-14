"""Forward and inverse kinematics of the arms other than SO101, outside the simulator.

The model is the preset's MuJoCo file under aetos_multiam/assets/mjcf. Poses are 4x4, of the
end-effector link in the drone's base frame; joint angles are the simulator's, in its order.

    kin = ArmKinematics("amov950_airbot_bottom")
    T = kin.fk(q)
    sols = kin.ik(T, q0=q)        # every solution found, nearest to q0 first; [] if none

The 6-joint AIRBOT Play arm is solved analytically (all elbow / wrist branches, from DISCOVERSE)
and each branch is then refined numerically; the other arms are solved numerically from several
starting postures. Arms with fewer than six joints cannot reach an arbitrary pose: pass
``position_only=True`` to ask for the position alone.

Run in the ``armplan`` env (needs mujoco).  Self-check: python scripts/arm_planning/arm_kinematics.py
"""
import os.path as osp

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

MJCF_DIR = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "aetos_multiam", "assets", "mjcf")
_AIRBOT = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# preset -> model file, drone base body, end-effector body, arm joints (names as in the simulator)
ARMS = {
    "amov950_airbot_bottom": dict(mjcf="amov950_airbot_final/robot.xml", base="quadrotor_950", ee="link6",
                                  joints=_AIRBOT, analytic_base="arm_base"),
    "amov950_airbot_dexterous_bottom": dict(mjcf="amov950_airbot_final/robot.xml", base="quadrotor_950", ee="link5",
                                            joints=_AIRBOT[:5]),
    "uaquad600_gripper": dict(mjcf="ambench_quad_600_top_arm_final/robot.xml", base="base_link", ee="ee_link",
                              joints=["arm_link1_yaw_joint", "arm_link2_pitch_joint", "arm_link3_pitch_joint",
                                      "arm_link4_pitch_joint"]),
    "fahexa750_gripper": dict(mjcf="ambench_fa_hexa_750_final/robot.xml", base="base_link", ee="ee_link",
                              joints=["arm_link1_pitch_joint", "arm_link2_pitch_joint", "arm_link3_pitch_joint",
                                      "arm_link4_roll_joint"]),
    # The arm-only file: the full robot.xml has a soft gripper this MuJoCo version does not load.
    "hummingbird340_2dof_gripper": dict(mjcf="hummingbird_2r_three_finger_50pct/dynamixel_2r_arm.xml",
                                        base="arm_carrier", ee="tool_adapter", joints=["R1", "R2"]),
}


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


class ArmKinematics:
    def __init__(self, preset: str, limits=None):
        """``limits``: (n, 2) joint limits to use instead of the model file's (e.g. the simulator's)."""
        cfg = ARMS[preset]
        self.m = mujoco.MjModel.from_xml_path(osp.join(MJCF_DIR, cfg["mjcf"]))
        self.d = mujoco.MjData(self.m)
        self.joint_names, self.n = list(cfg["joints"]), len(cfg["joints"])
        self.adr = np.array([self.m.joint(j).qposadr[0] for j in cfg["joints"]])
        self.dof = np.array([self.m.joint(j).dofadr[0] for j in cfg["joints"]])
        self.base, self.ee = self.m.body(cfg["base"]).id, self.m.body(cfg["ee"]).id
        self.limits = np.array([self.m.joint(j).range for j in cfg["joints"]]) if limits is None else np.asarray(limits, float)
        self.analytic_base = self.m.body(cfg["analytic_base"]).id if "analytic_base" in cfg else None

    def _pose(self, body):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.d.xmat[body].reshape(3, 3), self.d.xpos[body]
        return T

    def _set(self, q):
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # the Jacobian needs it

    def fk(self, q, body=None):
        """End-effector (or the named body's) pose in the drone base frame."""
        self._set(q)
        return _inv(self._pose(self.base)) @ self._pose(self.ee if body is None else self.m.body(body).id)

    def error(self, q, target, position_only=False):
        """(position error m, orientation error rad) of posture ``q`` against ``target``."""
        T = self.fk(q)
        rot = 0.0 if position_only else float(np.linalg.norm(R.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec()))
        return float(np.linalg.norm(target[:3, 3] - T[:3, 3])), rot

    def _refine(self, q, target, position_only, iters=200, damping=1e-2, lever=0.2):
        """Damped least-squares iterations from ``q``, inside the joint limits."""
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        q = np.clip(np.asarray(q, float), lo, hi)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            T = self.fk(q)
            Rb = self._pose(self.base)[:3, :3]
            mujoco.mj_jacBody(self.m, self.d, jp, jr, self.ee)
            e, J = target[:3, 3] - T[:3, 3], Rb.T @ jp[:, self.dof]
            if not position_only:  # a radian of orientation counts as ``lever`` metres
                e = np.r_[e, lever * R.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec()]
                J = np.r_[J, lever * Rb.T @ jr[:, self.dof]]
            if np.linalg.norm(e) < 1e-9:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
            q = np.clip(q + dq * min(1.0, 0.5 / max(np.abs(dq).max(), 1e-12)), lo, hi)
        return q

    def _analytic_seeds(self, target):
        from airbot_play_ik import AirbotPlayIK

        self._set(np.zeros(self.n))  # arm base in the drone base frame does not depend on q
        T = _inv(_inv(self._pose(self.base)) @ self._pose(self.analytic_base)) @ target
        try:
            return [np.array(s) for s in AirbotPlayIK().inverseKin(T[:3, 3], T[:3, :3])]
        except ValueError:  # it raises when it finds nothing
            return []

    def ik(self, target, q0=None, position_only=False, restarts=16, pos_tol=1e-4, rot_tol=1e-3, seed=0):
        """Postures that put the end-effector at ``target`` (4x4, drone base frame).

        Returns the distinct solutions found, nearest to ``q0`` first (or in the order found), each
        within ``pos_tol`` m and, unless ``position_only``, ``rot_tol`` rad. Empty if none.
        """
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        rng = np.random.default_rng(seed)
        seeds = [] if q0 is None else [np.asarray(q0, float)]
        if self.analytic_base is not None and not position_only:
            seeds += self._analytic_seeds(target)
        seeds += [rng.uniform(lo, hi) for _ in range(restarts)]
        sols = []
        for s in seeds:
            q = self._refine(s, target, position_only)
            ep, er = self.error(q, target, position_only)
            if ep < pos_tol and er < rot_tol and not any(np.abs(q - o).max() < 1e-3 for o in sols):
                sols.append(q)
        if q0 is not None:
            sols.sort(key=lambda q: np.abs(q - q0).sum())
        return sols


if __name__ == "__main__":  # round trip: the pose of a random posture must be solvable, and solutions must reproduce it
    rng = np.random.default_rng(0)
    for preset in ARMS:
        kin = ArmKinematics(preset)
        lo, hi = np.clip(kin.limits[:, 0], -3, 3), np.clip(kin.limits[:, 1], -3, 3)
        position_only = kin.n < 5  # a 4- or 2-joint arm is asked for position only
        found, count, worst = 0, [], 0.0
        for _ in range(50):
            q = rng.uniform(lo + 0.1, hi - 0.1)
            target = kin.fk(q)
            sols = kin.ik(target, q0=q + rng.uniform(-0.3, 0.3, kin.n), position_only=position_only)
            found += bool(sols)
            count.append(len(sols))
            for s in sols:
                assert (s >= kin.limits[:, 0] - 1e-9).all() and (s <= kin.limits[:, 1] + 1e-9).all()
                worst = max(worst, kin.error(s, target, position_only)[0])
        print(f"{preset}: {kin.n} joints, {'position only' if position_only else 'full pose'}: solved {found}/50, "
              f"{np.mean(count):.1f} solutions per pose, worst position error {worst * 1000:.3f} mm")
        assert found >= 48, preset
    print("arm_kinematics self-check OK")
