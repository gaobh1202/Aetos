"""Drone-base state for the arm policy: only what the flight controller itself reports.

Tilt (the gravity direction's x, y in the body frame: roll and pitch), body-frame linear velocity
and yaw rate. Left out on purpose: position and heading, which a policy could only use relative
to the object (privileged), and roll / pitch rates, which the simulated base never has.
Shared by export_arm_grasp_lerobot.py (training) and collect_arm_grasp_demo.py --policy (test).
"""
import numpy as np

NAMES = ["base_gx", "base_gy", "base_vx", "base_vy", "base_vz", "base_wz"]


def base_state(prev, cur, dt):
    """prev, cur: 4x4 world poses of base_link, dt seconds apart."""
    prev, cur = np.asarray(prev, float), np.asarray(cur, float)
    rot, rel = cur[:3, :3], prev[:3, :3].T @ cur[:3, :3]
    return np.array([-rot[2, 0], -rot[2, 1], *(rot.T @ (cur[:3, 3] - prev[:3, 3]) / dt),
                     (rel[1, 0] - rel[0, 1]) / (2 * dt)], dtype=np.float32)


if __name__ == "__main__":  # self-check: yawed 90 deg, moving along world y at 1 m/s, turning at 0.5 rad/s
    def pose(yaw, y):
        m = np.eye(4)
        m[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        m[1, 3] = y
        return m
    s = base_state(pose(np.pi / 2 - 0.005, 0.0), pose(np.pi / 2, 0.01), 0.01)
    assert np.allclose(s, [0, 0, 1, 0, 0, 0.5], atol=1e-4), s  # world y is the body's x
    print("ok")
