"""Frames shared by the SO101 arm-planning scripts (numpy only, no Isaac).

Chain: world -> drone base_link -> arm base (URDF ``base_link``) -> ... ->
``gripper_link`` -> grasp point. All quaternions wxyz. Joint angles passed to the
URDF are ``q_sim + Q_URDF_MINUS_SIM``.
"""
import glob
import os.path as osp

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
URDF_DIR = osp.join(ROOT, "aetos_multiam", "assets", "urdf", "so101")
URDF = osp.join(URDF_DIR, "so101_new_calib.urdf")
DEMO_DIR = osp.join(ROOT, "aetos_multiam", "data", "aerial_transport", "episodes")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Same numbers as manipulators/so_arm100.py (mount) and collect_aerial_transport_demo.py (grasp).
MOUNT_T, MOUNT_Q = (0.0, 0.02079, -0.023758238544791), (0.0, 0.7071067811865476, -0.7071067811865476, 0.0)
GRASP_LOCAL = (0.0108, -0.0004, -0.0682)  # handle-bar axis in the gripper_link frame
HANDLE_OFFSET_Z_AXIS = (0.0, 1.0, 0.0)  # bar runs along the box's y axis


def T(pos, quat_wxyz):
    m = np.eye(4)
    m[:3, :3] = R.from_quat(np.roll(np.asarray(quat_wxyz, float), -1)).as_matrix()
    m[:3, 3] = pos
    return m


# The simulated arm (Lightwheel so101_follower.usd) has the kinematics of LeRobot's
# so101_new_calib.urdf, but its root body sits elsewhere and its wrist_roll zero lacks
# the URDF's 0.0487 rad calibration tweak. Measured from PhysX link poses at 12 random
# configurations (agreement after correction < 0.001 mm):
T_USDBASE_URDFBASE = T((0.02079, 0.01576, 0.03248), (0.7071067811865476, 0.0, 0.0, -0.7071067811865476))
Q_URDF_MINUS_SIM = np.array([0.0, 0.0, 0.0, 0.0, 0.0486795])  # q_urdf = q_sim + this

T_DRONE_ARMBASE = T(MOUNT_T, MOUNT_Q) @ T_USDBASE_URDFBASE  # drone base_link -> URDF base_link
T_GRIPPER_GRASP = T(GRASP_LOCAL, (1, 0, 0, 0))


def load_demo(i=0):
    return np.load(sorted(glob.glob(osp.join(DEMO_DIR, "*.npz")))[i], allow_pickle=True)


def close_step(d):
    """First step of the 'close' phase: arm in place, jaw still open."""
    return int(np.argmax(d["phase"] == list(d["phase_names"]).index("close")))


def nominal_grasp(d, agent=0):
    """The demo's grasp, at the moment the jaw starts closing.

    Returns (T_world_armbase, T_world_handle, T_handle_grasp, q_default_sim):
    the handle frame sits at the bar centre with the box's axes (bar along y);
    T_handle_grasp is the hand-object pose every planned grasp must reproduce;
    q_default_sim is the arm command held through the demo (all zeros).
    """
    import placo  # only the planning env has it

    k, a = close_step(d), f"a{agent}"
    box = T(d["box_pos"][k], d["box_quat"][k])
    # Bar centre in the box frame: collect_aerial_transport_demo.py / CarryBox.handle_local_offsets.
    handle = box @ T(((-1.0, 1.0)[agent] * 0.42, 0.0, 0.34), (1, 0, 0, 0))
    armbase = T(d[f"{a}_drone_pos"][k], d[f"{a}_drone_quat"][k]) @ T_DRONE_ARMBASE
    robot = placo.RobotWrapper(osp.join(URDF_DIR, "so101_grasp.urdf"))
    for j, v in zip(ARM_JOINTS, d[f"{a}_arm_q"][k, :5] + Q_URDF_MINUS_SIM):
        robot.set_joint(j, float(v))
    robot.update_kinematics()
    grasp = armbase @ robot.get_T_world_frame("grasp_frame")
    return armbase, handle, np.linalg.inv(handle) @ grasp, d[f"{a}_arm_target"][k, :5].astype(float)
