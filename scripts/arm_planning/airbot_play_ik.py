# Analytic inverse kinematics of the AIRBOT Play arm, from DISCOVERSE
# (https://github.com/TATP-233/DISCOVERSE, discoverse/robots/airbot_play/airbot_play_ik.py,
# MIT License, Copyright (c) 2025 Yufei Jia), including the local wrist-singularity handling.
# Vendored unchanged below this header. Use it through arm_kinematics.ArmKinematics, which
# checks every returned solution against forward kinematics.
import numpy as np

class AirbotPlayIK:
    bias = np.array([0.0, -2.7549, 2.7549, 1.5708, 0.0, 0.0])
    a1 = 0.1172
    a3 = 0.27009
    a4 = 0.29015
    a6 = 0.23645
    arm_joint_range = np.array([
        [-3.151 , -2.963 , -0.094, -3.012 , -1.859 , -3.017 ],
        [ 2.089 ,  0.181 ,  3.161,  3.012 ,  1.859 ,  3.017 ]
    ])
    joint_range_scale = arm_joint_range[1] - arm_joint_range[0]

    arm_rot_mat = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    def __init__(self) -> None:
        pass

    def properIK(self, pos, ori, ref_q=None):
        return self.inverseKin(pos, ori @ self.arm_rot_mat, ref_q)

    def inverseKin(self, pos, ori, ref_q=None):
        assert len(pos) == 3 and ori.shape == (3,3)
        pos = self.move_joint6_2_joint5(pos, ori)
        angle = [0.0] * 6
        ret = []

        for i1 in [1, -1]:
            angle[0] = np.arctan2(i1 * pos[1], i1 * pos[0])
            c3 = (pos[0] ** 2 + pos[1] ** 2 + (pos[2] - self.a1) ** 2 - self.a3 ** 2 - self.a4 ** 2) / (2 * self.a3 * self.a4)
            if c3 > 1 or c3 < -1:
                raise ValueError("Fail to solve inverse kinematics: pos={}, ori={}".format(pos, ori))

            for i2 in [1, -1]:
                s3 = i2 * np.sqrt(1 - c3 ** 2)
                angle[2] = np.arctan2(s3, c3)
                k1 = self.a3 + self.a4 * c3
                k2 = self.a4 * s3
                angle[1] = np.arctan2(k1 * (pos[2] - self.a1) - i1 * k2 * np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                                   i1 * k1 * np.sqrt(pos[0] ** 2 + pos[1] ** 2) + k2 * (pos[2] - self.a1))
                R = np.array([
                    [np.cos(angle[0]) * np.cos(angle[1] + angle[2]),
                     -np.cos(angle[0]) * np.sin(angle[1] + angle[2]),
                     np.sin(angle[0])],
                    [np.sin(angle[0]) * np.cos(angle[1] + angle[2]),
                     -np.sin(angle[0]) * np.sin(angle[1] + angle[2]),
                     -np.cos(angle[0])],
                    [np.sin(angle[1] + angle[2]), np.cos(angle[1] + angle[2]), 0]
                ])
                ori1 = R.T @ ori
                wrist_radius = np.sqrt(ori1[2, 2] ** 2 + ori1[1, 2] ** 2)
                if wrist_radius < 0.25:
                    # joint5 ~ 0: joints 4 and 6 become (near-)coaxial (the
                    # exact wrist singularity at wrist_radius==0 -- see
                    # ARM_NEUTRAL_QPOS's docstring in
                    # live_visionpro_to_airbot_inspire.py for the full
                    # symptom description: joint4/joint6 fighting each
                    # other in the null space). The usual
                    # arctan2(i5*ori1[2,2], i5*ori1[1,2]) formula for
                    # angle[3] divides two small numbers here -- numerically
                    # unstable (ordinary pose noise can swing the ratio of
                    # two near-zero numbers by up to pi between adjacent
                    # calls), and this instability isn't limited to a
                    # razor-thin band at EXACTLY wrist_radius=0: it's a
                    # continuous sensitivity that's already severe well
                    # before that (confirmed empirically in this project
                    # before this fix -- a 0.2rad bias away from the exact
                    # singularity only cut jitter from ~61deg to ~12deg,
                    # not to zero). A hard cutoff at some tiny epsilon
                    # (e.g. 1e-3) would almost never trigger in practice
                    # and wouldn't fix the actual observed symptom.
                    #
                    # 0.25 is chosen from the OTHER side of that trade-off:
                    # only angle[3]+angle[5] (not each individually) is
                    # EXACTLY determined by the target orientation at
                    # wrist_radius==0; away from it the "split based on
                    # ref_q" approach below is an approximation with a
                    # small, smoothly-growing orientation error (measured
                    # against this project's own MuJoCo model: ~0.64deg at
                    # wrist_radius=0.25 i.e. joint5~14.3deg, ~2.6deg at
                    # joint5~28.6deg) -- negligible next to the up-to-180deg
                    # single-step JUMPS the unstable formula produces in
                    # the same region (isolated A/B test, 200-step sweep
                    # through joint5 in [-0.15,0.15]rad: old max jump
                    # 180.5deg/mean 6.96deg vs this branch's max 0.54deg/
                    # mean 0.34deg, zero >5deg steps either way).
                    #
                    # Ported from mmk2_kdl_py's ArmKdl.inverse_kinematics
                    # (the official airbot_vr/pico teleop SDK's analytic
                    # IK, 机械臂灵巧操作套装/airbot_vr_1.0.0) and re-derived/
                    # verified against THIS project's own MuJoCo model
                    # rather than translated blindly (that package's DH
                    # convention differs from this file's hand-derived
                    # one): at wrist_radius -> 0, ori1[2,0] = cos(angle[3]
                    # + angle[5]) and ori1[2,1] = sin(angle[3] + angle[5]),
                    # independent of the individual angle[3]/angle[5]
                    # split (confirmed numerically across many (angle[3],
                    # angle[5]) pairs sharing the same sum).
                    angle[4] = np.arctan2(wrist_radius, ori1[0, 2])
                    combined = np.arctan2(ori1[2, 1], ori1[2, 0])
                    if ref_q is not None:
                        ref_sum = ref_q[3] + ref_q[5]
                        # bring `combined` to whichever 2*pi branch is
                        # nearest ref_sum BEFORE splitting, so the even
                        # split below can't introduce a spurious +-2*pi
                        # jump relative to the arm's current pose
                        combined = combined + 2 * np.pi * np.round((ref_sum - combined) / (2 * np.pi))
                        half_delta = (combined - ref_sum) / 2
                        angle[3] = ref_q[3] + half_delta
                        angle[5] = ref_q[5] + half_delta
                    else:
                        angle[3] = combined / 2
                        angle[5] = combined / 2
                    js = self.add_bias(angle)
                    if np.all((js > self.arm_joint_range[0]) * (js < self.arm_joint_range[1])):
                        ret.append(js)
                else:
                    for i5 in [1, -1]:
                        angle[3] = np.arctan2(i5 * ori1[2, 2], i5 * ori1[1, 2])
                        angle[4] = np.arctan2(i5 * wrist_radius, ori1[0, 2])
                        angle[5] = np.arctan2(-i5 * ori1[0, 0], -i5 * ori1[0, 1])
                        js = self.add_bias(angle)
                        if np.all((js > self.arm_joint_range[0]) * (js < self.arm_joint_range[1])):
                            ret.append(js)
        if len(ret) == 0:
            raise ValueError("Fail to solve inverse kinematics: pos={}, ori={}".format(pos, ori))

        if ref_q is not None:
            joint_dist_lst = []
            for js in ret:
                joint_dist_lst.append(np.sum(np.abs(ref_q - js) / self.joint_range_scale))
            q = ret[np.argmin(joint_dist_lst)]
            return q
        else:
            return ret

    def add_bias(self, angle):
        ret = []
        for i in range(len(angle)):
            a = angle[i] + self.bias[i]
            while a > np.pi:
                a -= 2 * np.pi
            while a < -np.pi:
                a += 2 * np.pi
            ret.append(a)
        return ret

    def move_joint6_2_joint5(self, pos, ori):
        ret = np.array([
            -ori[0, 2] * self.a6 + pos[0],
            -ori[1, 2] * self.a6 + pos[1],
            -ori[2, 2] * self.a6 + pos[2]
        ])
        return ret

    def j3_ik(self, pos):
        angle = [0.0] * 3
        ret = []

        for i1 in [1, -1]:
            angle[0] = np.arctan2(i1 * pos[1], i1 * pos[0])
            c3 = (pos[0] ** 2 + pos[1] ** 2 + (pos[2] - self.a1) ** 2 - self.a3 ** 2 - self.a4 ** 2) / (2 * self.a3 * self.a4)
            if c3 > 1 or c3 < -1:
                raise ValueError("Fail to solve inverse kinematics")

            for i2 in [1, -1]:
                s3 = i2 * np.sqrt(1 - c3 ** 2)
                angle[2] = np.arctan2(s3, c3)
                k1 = self.a3 + self.a4 * c3
                k2 = self.a4 * s3
                angle[1] = np.arctan2(k1 * (pos[2] - self.a1) - i1 * k2 * np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                                   i1 * k1 * np.sqrt(pos[0] ** 2 + pos[1] ** 2) + k2 * (pos[2] - self.a1))
                js = self.add_bias(angle)
                if np.all((js > self.arm_joint_range[0,:3]) * (js < self.arm_joint_range[1,:3])):
                    ret.append(js)
        return ret

if __name__ == "__main__":
    arm_ik = AirbotPlayIK()

    trans = np.array([ 0.276, -0., 0.219])
    rot   = np.array([
        [1., -0., -0.],
        [0.,  1., -0.],
        [0.,  0.,  1.]
    ])

    qs = arm_ik.properIK(trans, rot)

    for q in qs:
        print(q)
