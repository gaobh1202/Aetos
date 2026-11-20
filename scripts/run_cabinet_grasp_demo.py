"""Scripted execution of AerialCabinetGrasp: two fahexa750_gripper pull the
cabinet's doors open by their handles, then the hummingbird340_2dof_gripper
flies into the opening, takes the button off its shelf and carries it out.

    python scripts/run_cabinet_grasp_demo.py
    python scripts/run_cabinet_grasp_demo.py --video outputs/aerial_cabinet_grasp/demo.mp4

Prints one line per phase and ``SUCCESS=True/False`` at the end; exits non-zero
on failure. ``--out`` (default ``outputs/aerial_cabinet_grasp/demo.npz``) gets
the whole trajectory: per-agent state, setpoint, rotor action and arm targets,
the doors' open fraction and the button's position.

Order of play. The two door handles are 10 cm apart with the doors shut and a
hexacopter is 1.1 m across, so the doors are opened one after the other: agent
0 takes the -x door and swings out to the left with it, then agent 1 does the
same on the right. Both stay where they end up, holding their door, while
agent 2 goes in between them.

Physics setup -- the simplified plant this is run on, to be dropped once a
whole-body controller exists:

* **Arm dynamics frozen first** (``freeze_arm_dynamics``, before anything
  moves): each arm link keeps only ``arm_dynamics_residual`` of its mass and
  inertia and the rest is lumped onto ``base_link``, so moving an arm no longer
  kicks the airframe it hangs from. The residual is each preset's own (0.1 for
  both robots here); it cannot simply be 0, the solver diverges long before
  that -- see ``utils/arm_dynamics.py``. ``--arm-residual`` overrides it.
* **Each drone is flown by a PID position controller**
  (``controllers/drone/pid_position_controller.py``) on a scripted setpoint,
  every control step: 6-DOF wrench allocation underneath for the fully-actuated
  hexacopters, so they translate and pull without tilting; the Lee attitude
  loop underneath for the Hummingbird. The integral term is what holds a
  gripper on a handle against the hinge's resistance.
* Arm joints are PD-driven to fixed postures: the hexacopters' arm to the one
  that puts the gap between its fingers ``HANDLE_STANDOFF_M`` ahead of the
  airframe and ``HANDLE_BELOW_M`` above it with the gripper level (solved from
  the arm's link geometry), the Hummingbird's straight down. The drone, not
  the arm, does the positioning: its setpoint is corrected every step by where
  the grasp point actually is relative to the airframe.
* **Grasps are geometric, and what is grasped is then moved kinematically.**
  Gripper <-> door and gripper <-> button contacts are filtered
  (``task.filter_grasp_contacts``): with the arm frozen a finger weighs a few
  grams, and such a finger does not hold a 5 kg door, it gets shoved around
  by it. Instead the fingers close to the width of what they are closing on,
  the grasp is accepted only if the grasp point is on the handle (or the
  button cap) within tolerance, and from then on
  - a held door's hinge angle is set every step to the angle the gripper is
    at about that hinge, so the door goes exactly where the gripper takes it;
    how far the gripper strays off the arc the handle can physically follow
    is tracked, and past ``DOOR_SLIP_TOL_M`` the run fails;
  - the hinge's resistance (its damping, at the rate the door is being
    turned) is applied to the pulling drone as a force on ``base_link``, so
    the controller does have something to pull against;
  - the button is carried at the pose it had in the gripper's frame when the
    fingers closed.
  Not simulated: the closed kinematic chain drone-arm-door, finger friction,
  the button's weight on the Hummingbird (it is 4 cm across).
"""
import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default=str(REPO / "outputs/aerial_cabinet_grasp/demo.npz"),
                    help="Where the recorded trajectory goes.")
parser.add_argument("--video", default=None, help="Optional: write an .mp4 of the run (overview camera).")
parser.add_argument("--video-every", type=int, default=5, help="Capture one frame every N control steps.")
parser.add_argument("--arm-residual", type=float, default=None,
                    help="Override every preset's arm_dynamics_residual (1.0 = arm dynamics not frozen).")
parser.add_argument("--ki", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Integral gain of every drone's position PID, instead of the per-role defaults. "
                         "0 0 0 flies the plain PD loop, for comparison.")
parser.add_argument("--only", choices=["all", "doors", "button"], default="all",
                    help="Run one half of the task: 'doors' stops once both doors are open; 'button' starts "
                         "with the doors already open and unattended, and only the Hummingbird works.")
parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim window instead of headless.")
args = parser.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

#: 100 Hz physics and control -- what examples/show_am_action.py flies these two
#: airframes at.
DT = 0.01

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=[
        "task=AerialCabinetGrasp", f"headless={str(not args.gui).lower()}",
        "++task.env.num_envs=1", "++task.env.max_episode_length=100000",
        "++task.filter_grasp_contacts=true",
        f"++task.sim.dt={DT}", "++task.sim.substeps=1",
    ])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
simulation_app = init_simulation_app(cfg)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchrl.envs.utils import step_mdp  # noqa: E402

from omni_drones.views import RigidPrimView  # noqa: E402

from aetos_multiam.controllers import make_pid_position_controller  # noqa: E402
from aetos_multiam.environments import make as make_env  # noqa: E402
from aetos_multiam.environments.collaborative.aerial_cabinet_grasp import (  # noqa: E402
    HANDLE_BELOW_M, HANDLE_STANDOFF_M, NUM_DOOR_AGENTS, RETRIEVER_AGENT,
)
from aetos_multiam.utils import freeze_arm_dynamics  # noqa: E402

# -- fahexa750 arm: planar 3R + wrist roll, in base_link's x-z plane -----------
# Link geometry read off data/arm_kinematics/fahexa750_gripper.npz (24 postures
# recorded from the simulator): each link's origin in its parent's frame, and
# the sign with which each joint angle turns it about base_link's +y.
HEXA_SHOULDER = np.array([0.088, 0.0647])         # arm_link1_pitch_link, in base_link (x, z)
HEXA_UPPER = np.array([-0.3795, 0.059])           # link1 -> link2
HEXA_FOREARM = 0.4475                             # link2 -> link3, along link2's x
HEXA_WRIST = 0.071                                # link3 -> ee_link, along link3's x
#: The grasp point, in ee_link's frame: on the axis between the two fingers,
#: which run from x = 0.074 to 0.194 m. The handle stands 74 mm proud of the
#: door and its centre is taken half way out, so at 0.158 the fingertips stop
#: 1 mm short of the door's face and the handle's outer face is 25 mm clear of
#: the palm (which ends at x = 0.096).
HEXA_GRASP_LOCAL = (0.158, 0.0, 0.0)
#: How far each finger is drawn back to go over a handle: a 70 mm gap, 12 mm
#: of room either side of the 46 mm bar. Not the full 90 mm the fingers can
#: do -- the two handles are 10 cm apart with the doors shut, and a gripper
#: opened that wide on one of them has its outer edge 1 cm from the other.
HEXA_FINGER_OPEN = 0.035
# -- hummingbird340 gripper ---------------------------------------------------
#: The grasp point, in the gripper Housing's frame: on the tool axis, 15 mm
#: short of the fingertips (which reach z = 0.105..0.111 m).
TOOL_GRASP_LOCAL = (0.0, 0.0, 0.090)
#: Finger angle at which the three fingertips stand on a 21 mm radius -- the
#: button cap's. Read off the finger collision meshes at recorded postures:
#: tip radius 39 mm at 0.08 rad, 20 mm at 0.46, 12 mm at 0.59.
TOOL_FINGER_ON_CAP = 0.44
#: Height of the cap's centre above the button's own origin (its base).
BUTTON_CAP_Z_M = 0.066

PREGRASP_BACKOFF_M = 0.30      # how far short of the handle a hexacopter lines up
GRASP_TOL_M = 0.010            # grasp point to handle axis, when the fingers close: inside
                               # the room the open fingers leave around the bar
DOOR_SLIP_TOL_M = 0.03         # gripper off the handle's arc while pulling
#: How far each door is swung, as a fraction of its travel: about 88 deg, short
#: of the hinge's stop. Swung to the stop itself, the last of the integral term
#: -- still pushing against a hinge that has stopped resisting -- carried the
#: gripper 3 cm past where the door can follow (measured).
DOOR_OPEN_TARGET = 0.95
BUTTON_TOL_M = 0.012           # tool axis to button axis, when the fingers close
ENTRY_CLEARANCE_M = 0.10       # how far above grasp height the Hummingbird goes in
STAGING_M = 1.3                # how far out from the cabinet front it lines up
DELIVERY_XYZ = (0.0, -2.5, 1.3)
G_BODY_ALPHA = 0.05            # low-pass on the measured grasp-point offset
#: Integral gain for the two door agents, twice the controller's default: the
#: hinge pushes back with 2-3 N for as long as the door is moving, and at 1.5
#: the term took most of the swing to build up to it (7 cm of position error
#: at mid-swing, measured). Still well inside Kp * Kd = 8.8, the stability bound.
DOOR_AGENT_KI = (3.0, 3.0, 2.0)
SP_ACC_ALPHA = 0.05            # low-pass on the setpoint's own acceleration (feedforward)
SP_ACC_MAX = 1.0               # ...and its bound, m/s^2

PHASES = [("settle", 3.0)]
if args.only != "button":
    for _d in range(NUM_DOOR_AGENTS):
        PHASES += [(f"fly_{_d}", 6.0), (f"align_{_d}", 1.5), (f"approach_{_d}", 4.0),
                   (f"seat_{_d}", 2.0), (f"close_{_d}", 1.5), (f"open_{_d}", 16.0)]
if args.only != "doors":
    PHASES += [("retriever_fly", 6.0), ("retriever_enter", 6.0), ("retriever_descend", 4.0),
               ("retriever_close", 1.5), ("retriever_lift", 2.0), ("retriever_exit", 6.0),
               ("retriever_deliver", 4.0)]
PHASES += [("hold", 2.0)]


def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def yaw_rot(yaw: float, v) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# Quaternions are (w, x, y, z) throughout, as numpy: every pose the script
# reasons about is read off the GPU once per step and worked on here.
def q_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def q_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_rot(q, v) -> np.ndarray:
    return q_mul(q_mul(q, np.array([0.0, *v])), q_conj(q))[1:]


def q_yaw(q) -> float:
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


def q_up_z(q) -> float:
    """z-component of the body's up axis: cos(tilt)."""
    return 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)


def hexa_arm_posture(forward: float, up: float, q1_range) -> np.ndarray:
    """Joint angles (pitch1, pitch2, pitch3, roll) that put the grasp point
    ``forward`` ahead of base_link and ``up`` above it with the gripper level.

    Level means link3's pitch, q1 - q2 + q3, is zero, which fixes the wrist at
    (forward - wrist - grasp offset, up); what is left is a two-link problem.
    Of its two solutions the elbow-up one is taken: the other folds the
    forearm down through the rotor plane.
    """
    wrist = np.array([forward - HEXA_WRIST - HEXA_GRASP_LOCAL[0], up]) - HEXA_SHOULDER

    def elbow(q1):                      # link2's origin relative to the shoulder
        c, s = math.cos(q1), math.sin(q1)
        return np.array([HEXA_UPPER[0] * c + HEXA_UPPER[1] * s, -HEXA_UPPER[0] * s + HEXA_UPPER[1] * c])

    def gap(q1):
        return np.linalg.norm(wrist - elbow(q1)) - HEXA_FOREARM

    grid = np.linspace(q1_range[0], q1_range[1], 2000)
    values = np.array([gap(q) for q in grid])
    best = None
    for k in np.nonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0]:
        lo, hi = grid[k], grid[k + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.sign(gap(mid)) == np.sign(gap(lo)) else (lo, mid)
        q1 = 0.5 * (lo + hi)
        d = wrist - elbow(q1)
        pitch2 = math.atan2(-d[1], d[0])          # link2's pitch = q1 - q2
        if best is None or elbow(q1)[1] > best[0]:
            best = (elbow(q1)[1], np.array([q1, q1 - pitch2, -pitch2, 0.0]))
    if best is None:
        raise RuntimeError(f"fahexa750 arm cannot reach {forward} m forward, {up} m up")
    return best[1]


try:
    env = make_env("AerialCabinetGrasp", cfg, headless=not args.gui)
    td = env.reset()
    dev = env.device
    dt = float(env.sim.get_physics_dt()) * env.substeps
    origin = env.envs_positions[0].cpu().numpy()
    cabinet, button = env.cabinet, env.button
    env_ids = torch.arange(env.num_envs, device=dev)
    keys = env._agent_keys

    if args.only == "button":
        cabinet.reset_doors(env_ids, open_fraction=1.0)

    # ---- 1. take the arm out of the airframe's dynamics, before anything moves --
    agents = []
    for i, am in enumerate(env.aerial_manipulators):
        d = am.drone
        residual = am.cfg.arm_dynamics_residual if args.arm_residual is None else args.arm_residual
        # Once per process: freeze_arm_dynamics is not idempotent.
        frozen = freeze_arm_dynamics(am, residual=float(residual))
        ki = args.ki if args.ki is not None else (DOOR_AGENT_KI if i < NUM_DOOR_AGENTS else None)
        pid_kwargs = {} if ki is None else {"ki": ki}
        if i < NUM_DOOR_AGENTS:
            # What these two fight comes in through the arm and turns with them.
            pid_kwargs["integral_frame"] = "heading"
        # On the CPU: it is a few 3-vectors per step, and every tensor it would
        # otherwise hand back costs a GPU sync.
        ctrl = make_pid_position_controller(d, dt, **pid_kwargs).to("cpu")
        names = [n.split("/")[-1] for n in d._view.body_names]
        joint_names = am.teleop_joint_names
        limits = d._view.get_dof_limits()[0, 0][am.teleop_joint_indices].cpu().numpy()
        joint_idx = am.teleop_joint_indices.to(dev)
        a = {"am": am, "drone": d, "ctrl": ctrl, "names": names, "joint_names": joint_names,
             "limits": limits, "joint_idx": joint_idx, "load": np.zeros(3),
             "is_door_agent": i < NUM_DOOR_AGENTS}
        print(f"[demo] agent {i} ({am.cfg.drone_type}): arm dynamics frozen at residual {residual} "
              f"({frozen['arm_mass']:.3f} kg of arm, base_link now {frozen['base_mass']:.3f} kg); "
              f"{type(ctrl.inner).__name__} + I, kp={ctrl.kp.tolist()} ki={ctrl.ki.tolist()} "
              f"kd={ctrl.kd.tolist()}; joints {joint_names}", flush=True)

        # The hinge's resistance, as a force on base_link: wrap the one
        # base_link force write apply_action makes, so the thrust and torques it
        # computes stay untouched (a second apply call would replace them).
        if a["is_door_agent"]:
            orig_apply = d.base_link.apply_forces_and_torques_at_pos

            def patched(forces=None, torques=None, positions=None, is_global=True, _a=a, _orig=orig_apply):
                if np.any(_a["load"]) and forces is not None:
                    forces = forces + torch.as_tensor(_a["load"], device=dev, dtype=forces.dtype).reshape(1, 3)
                return _orig(forces, torques, positions, is_global=is_global)

            d.base_link.apply_forces_and_torques_at_pos = patched
        agents.append(a)

    def refresh() -> None:
        """Read the simulation once per step: each robot's root state, link
        poses and arm joints. Everything else below works on these copies --
        with the GPU shared, every separate read is a wait for a time slice."""
        for b in agents:
            v = b["drone"]._view
            b["st"] = b["drone"].get_state()[0, 0, :13].cpu().numpy().astype(np.float64)
            b["links"] = v._physics_view.get_link_transforms().reshape(
                -1, len(b["names"]), 7)[0].cpu().numpy().astype(np.float64)
            b["q"] = v.get_joint_positions()[0, 0][b["joint_idx"]].cpu().numpy().astype(np.float64)

    def drone_pos(a) -> np.ndarray:
        return a["st"][:3] - origin

    def drone_yaw(a) -> float:
        return q_yaw(a["st"][3:7])

    def link_pose(a, name):
        """(pos, quat wxyz) of one link, world frame. PhysX returns xyzw."""
        lt = a["links"][a["names"].index(name)]
        return lt[:3], np.array([lt[6], lt[3], lt[4], lt[5]])

    def grasp_point(a) -> np.ndarray:
        link, local = ("ee_link", HEXA_GRASP_LOCAL) if a["is_door_agent"] else ("Housing", TOOL_GRASP_LOCAL)
        p, q = link_pose(a, link)
        return p + q_rot(q, local) - origin

    def joint_targets(a, **by_name) -> np.ndarray:
        """Teleop-order target vector; joints not named keep their last target."""
        t = a.get("arm_target", np.zeros(len(a["joint_names"]))).copy()
        for name, value in by_name.items():
            t[a["joint_names"].index(name)] = value
        return np.clip(t, a["limits"][:, 0], a["limits"][:, 1])

    # ---- arm postures --------------------------------------------------------
    refresh()
    for i, a in enumerate(agents):
        a["arm_target"] = a["q"].copy()
        if a["is_door_agent"]:
            q1_range = a["limits"][a["joint_names"].index("arm_link1_pitch_joint")]
            q = hexa_arm_posture(HANDLE_STANDOFF_M, HANDLE_BELOW_M, q1_range)
            a["arm_work"] = joint_targets(
                a, arm_link1_pitch_joint=q[0], arm_link2_pitch_joint=q[1], arm_link3_pitch_joint=q[2],
                arm_link4_roll_joint=q[3], arm_lfinger_joint=HEXA_FINGER_OPEN, arm_rfinger_joint=HEXA_FINGER_OPEN)
            a["g_body"] = np.array([HANDLE_STANDOFF_M, 0.0, HANDLE_BELOW_M])
            print(f"[demo] agent {i}: arm posture for the handle (rad) {np.round(q, 3).tolist()}", flush=True)
        else:
            a["arm_work"] = joint_targets(a, R1=0.0, R2=0.0, finger_1=0.0, finger_2=0.0, finger_3=0.0)
            a["g_body"] = None                    # measured once the arm has settled
        a["sp"] = drone_pos(a)
        a["yaw_sp"] = drone_yaw(a)

    # ---- the cabinet, as the script sees it ------------------------------------
    bar_half = cabinet.scale * cabinet.handle_bar_section_m / 2.0
    bar_half_len = cabinet.scale * cabinet.handle_bar_length_m / 2.0
    hinge = [np.array(cabinet.to_parent((*cabinet.hinge_xy[dd], 0.0)))[:2] for dd in range(NUM_DOOR_AGENTS)]

    def handle(dd, f) -> np.ndarray:
        return np.array(cabinet.handle_position(dd, f))

    def normal(dd, f) -> np.ndarray:
        return np.array([*cabinet.handle_normal(dd, f), 0.0])

    def heading(dd, f) -> float:
        """Yaw that points a hexacopter's arm at the handle."""
        n = normal(dd, f)
        return math.atan2(-n[1], -n[0])

    def angle_about_hinge(dd, p) -> float:
        return math.atan2(p[1] - hinge[dd][1], p[0] - hinge[dd][0])

    arc = []
    for dd in range(NUM_DOOR_AGENTS):
        a0 = angle_about_hinge(dd, handle(dd, 0.0))
        arc.append({"closed": a0, "sweep": wrap(angle_about_hinge(dd, handle(dd, 1.0)) - a0),
                    "radius": float(np.linalg.norm(handle(dd, 0.0)[:2] - hinge[dd]))})
    hinge_damping = cabinet.hinge_damping * 180.0 / math.pi          # N*m*s/rad

    # ---- the button: all three of its bodies, so it can be carried as one ----
    button_views = [button._view]
    for body in button.root_body_names[1:]:
        view = RigidPrimView(f"{env.env_ns}/env_.*/{button.name}/{body}", reset_xform_properties=False)
        view.initialize(env.sim.physics_sim_view)
        button_views.append(view)

    def button_pose(view=None):
        """(pos, quat wxyz) of one of the button's bodies, world frame."""
        p, q = (view or button._view).get_world_poses()
        return (p.reshape(-1, 3)[0].cpu().numpy().astype(np.float64),
                q.reshape(-1, 4)[0].cpu().numpy().astype(np.float64))

    def cap_center() -> np.ndarray:
        return button_now - origin + np.array([0.0, 0.0, BUTTON_CAP_Z_M])

    button_now = button_pose()[0]
    doors_now = cabinet.get_door_open_fraction()[0].cpu().numpy()

    front = np.array(cabinet.front_normal)
    front_xy = np.array(cabinet.shelf_position(env.button_compartment, 0.5, 0.0))[:2]

    # ---- video ---------------------------------------------------------------
    frames_dir = vp = None
    if args.video:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf

        vp = get_active_viewport()
        cam = ViewportCameraState("/OmniverseKit_Persp", vp)
        ox, oy = float(origin[0]), float(origin[1])
        cam.set_position_world(Gf.Vec3d(ox + 3.4, oy - 5.6, 3.7), True)
        cam.set_target_world(Gf.Vec3d(ox + 0.0, oy + 0.1, 1.15), True)
        frames_dir = Path(args.video).with_suffix("")
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True)

    rec = {k: [] for k in ["t", "phase", "door_open", "door_held", "button_pos", "button_held"]}
    for i in range(len(agents)):
        for k in ["pos", "quat", "vel", "setpoint_pos", "setpoint_yaw", "rotor_action", "arm_target",
                  "arm_q", "grasp_point", "integral_acc"]:
            rec[f"a{i}_{k}"] = []

    door_held = [False] * NUM_DOOR_AGENTS
    door_cmd = [0.0] * NUM_DOOR_AGENTS           # open fraction the script is asking for
    door_slip = [0.0] * NUM_DOOR_AGENTS          # worst distance off the arc, metres
    pull_tilt = [0.0] * NUM_DOOR_AGENTS          # worst body tilt while swinging a door, degrees
    grasp_err = [float("nan")] * NUM_DOOR_AGENTS
    button_held, button_rel, button_err = False, None, float("nan")
    status, step, frame = "ok", 0, 0
    retriever = agents[RETRIEVER_AGENT]

    def track(a, target_gp, yaw_sp):
        """Setpoint that puts the agent's grasp point on ``target_gp``: where
        the grasp point actually sits relative to the airframe (low-passed, in
        the heading frame) is taken off the target.

        Turned back into the world with the heading the drone HAS, not the one
        it was asked for. The grasp point is 0.75 m out in front of a
        hexacopter, so 2 deg of yaw lag while it swings round with a door puts
        the gripper 3 cm off the handle sideways -- measured, with the
        commanded heading here. This way the position loop makes up for it.
        """
        yaw = drone_yaw(a)
        g = yaw_rot(-yaw, grasp_point(a) - drone_pos(a))
        a["g_body"] = g if a["g_body"] is None else (1 - G_BODY_ALPHA) * a["g_body"] + G_BODY_ALPHA * g
        return np.asarray(target_gp) - yaw_rot(yaw, a["g_body"])

    for pid, (pname, dur) in enumerate(PHASES):
        n = max(1, int(round(dur / dt)))
        kind, _, tail = pname.rpartition("_")
        dd = int(tail) if tail.isdigit() else None
        a = agents[dd] if dd is not None else retriever
        s_sp, s_yaw, s_arm, s_gp = a["sp"].copy(), a["yaw_sp"], a["arm_target"].copy(), grasp_point(a)
        cap0 = cap_center()

        if pname == "settle":
            snaps = [(b["arm_target"].copy(), b["arm_work"]) for b in agents]
        elif kind == "fly":
            yaw_goal = s_yaw + wrap(heading(dd, 0.0) - s_yaw)
        elif kind == "open" and not door_held[dd]:
            status = f"door{dd}_grasp_failed"
            break
        elif pname == "retriever_lift" and not button_held:
            status = "button_grasp_failed"
            break

        for k in range(n):
            u = min_jerk((k + 1) / n)
            if pname == "settle":
                # Everyone holds station while the arms move to their postures.
                for b, (q0, q1) in zip(agents, snaps):
                    b["arm_target"] = q0 + u * (q1 - q0)
            elif kind == "fly":
                goal = handle(dd, 0.0) + normal(dd, 0.0) * (HANDLE_STANDOFF_M + PREGRASP_BACKOFF_M)
                goal[2] -= HANDLE_BELOW_M
                a["sp"] = s_sp + u * (goal - s_sp)
                a["yaw_sp"] = s_yaw + u * (yaw_goal - s_yaw)
            elif kind in ("align", "approach"):
                back = PREGRASP_BACKOFF_M * (1.0 if kind == "align" else 1.0 - u)
                a["yaw_sp"] = heading(dd, 0.0)
                a["sp"] = track(a, handle(dd, 0.0) + normal(dd, 0.0) * back, a["yaw_sp"])
            elif kind == "seat":
                # Hold the open fingers around the bar until the error is gone.
                a["sp"] = track(a, handle(dd, 0.0), a["yaw_sp"])
            elif kind == "close":
                a["sp"] = track(a, handle(dd, 0.0), a["yaw_sp"])
                q = HEXA_FINGER_OPEN + u * (bar_half - HEXA_FINGER_OPEN)
                a["arm_target"] = joint_targets(a, arm_lfinger_joint=q, arm_rfinger_joint=q)
            elif kind == "open":
                f = door_cmd[dd] = DOOR_OPEN_TARGET * u
                a["yaw_sp"] = s_yaw + wrap(heading(dd, f) - heading(dd, 0.0))
                a["sp"] = track(a, handle(dd, f), a["yaw_sp"])
            elif pname == "retriever_fly":
                goal = np.array([*(front_xy + front * STAGING_M),
                                 env.button_hover_position[2] + ENTRY_CLEARANCE_M])
                a["sp"] = s_sp + u * (goal - s_sp)
            elif pname == "retriever_enter":
                # From here on the tool's grasp point is steered, not the body.
                target = cap0 + np.array([0.0, 0.0, ENTRY_CLEARANCE_M])
                a["sp"] = track(a, s_gp + u * (target - s_gp), a["yaw_sp"])
            elif pname == "retriever_descend":
                a["sp"] = track(a, cap0 + np.array([0.0, 0.0, ENTRY_CLEARANCE_M * (1.0 - u)]), a["yaw_sp"])
            elif pname == "retriever_close":
                a["sp"] = track(a, cap0, a["yaw_sp"])
                q = u * TOOL_FINGER_ON_CAP
                a["arm_target"] = joint_targets(a, finger_1=q, finger_2=q, finger_3=q)
            elif pname == "retriever_lift":
                a["sp"] = s_sp + u * np.array([0.0, 0.0, ENTRY_CLEARANCE_M])
            elif pname == "retriever_exit":
                goal = np.array([*(front_xy + front * STAGING_M), s_sp[2]])
                a["sp"] = s_sp + u * (goal - s_sp)
            elif pname == "retriever_deliver":
                a["sp"] = s_sp + u * (np.array(DELIVERY_XYZ) - s_sp)

            # ---- held doors go where their gripper is ---------------------------
            for j in range(NUM_DOOR_AGENTS):
                b = agents[j]
                b["load"][:] = 0.0
                if not door_held[j]:
                    continue
                gp = grasp_point(b)
                frac = wrap(angle_about_hinge(j, gp) - arc[j]["closed"]) / arc[j]["sweep"]
                frac = min(max(frac, 0.0), 1.0)
                slip = float(np.linalg.norm(gp - handle(j, frac)))
                door_slip[j] = max(door_slip[j], slip)
                if slip > DOOR_SLIP_TOL_M:
                    status = f"door{j}_slipped"
                rate = (frac - b["door_frac"]) / dt * arc[j]["sweep"]            # rad/s about the hinge
                b["door_frac"] = frac
                cabinet.set_door_open_fraction(env_ids, j, frac)
                # Damping torque over the handle's lever arm, against the motion.
                tangent = np.array([-(gp[1] - hinge[j][1]), gp[0] - hinge[j][0], 0.0]) / arc[j]["radius"]
                b["load"][:] = -hinge_damping * rate / arc[j]["radius"] * tangent

            # ---- a held button goes where the gripper is --------------------------
            if button_held:
                hp, hq = link_pose(retriever, "Housing")
                for view, (p_rel, q_rel) in zip(button_views, button_rel):
                    view.set_world_poses(
                        torch.as_tensor(hp + q_rot(hq, p_rel), device=dev, dtype=torch.float32).reshape(1, 3),
                        torch.as_tensor(q_mul(hq, q_rel), device=dev, dtype=torch.float32).reshape(1, 4))
                    view.set_velocities(torch.zeros(1, 6, device=dev))

            # ---- fly: PID position loop per drone, PD drives on the arm joints ----
            for i, (key, b) in enumerate(zip(keys, agents)):
                st = b["st"]
                # Setpoint velocity as feedforward: without it the D term treats
                # the setpoint's own motion as an error, and the drone trails a
                # moving setpoint by Kd/Kp seconds of travel (0.16 m, measured).
                # ...and its acceleration, low-passed and bounded, for the same
                # reason one order up: Kp alone leaves acc/Kp of lag, 2.5 cm on
                # the arc a door is pulled along.
                sp_vel = (b["sp"] - b.get("sp_prev", b["sp"])) / dt
                acc = np.clip((sp_vel - b.get("sp_vel_prev", sp_vel)) / dt, -SP_ACC_MAX, SP_ACC_MAX)
                b["sp_acc"] = (1 - SP_ACC_ALPHA) * b.get("sp_acc", np.zeros(3)) + SP_ACC_ALPHA * acc
                b["sp_prev"], b["sp_vel_prev"] = b["sp"].copy(), sp_vel
                cmd = b["ctrl"].compute(
                    torch.as_tensor(st, dtype=torch.float32).reshape(1, 13),
                    target_pos=torch.as_tensor(b["sp"] + origin, dtype=torch.float32).reshape(1, 3),
                    target_vel=torch.as_tensor(sp_vel, dtype=torch.float32).reshape(1, 3),
                    target_acc=torch.as_tensor(b["sp_acc"], dtype=torch.float32).reshape(1, 3),
                    target_yaw=torch.tensor([[b["yaw_sp"]]], dtype=torch.float32),
                ).clamp(-1.0, 1.0)
                arm_t = torch.as_tensor(b["arm_target"], device=dev, dtype=torch.float32)
                b["am"].hold_joints(arm_t.reshape(1, -1))
                action = torch.zeros(env.action_spec[key, "action"].shape, device=dev)
                action[..., : cmd.shape[-1]] = cmd.to(dev)
                td.set((key, "action"), action)

                rec[f"a{i}_pos"].append(drone_pos(b).astype(np.float32))
                rec[f"a{i}_quat"].append(st[3:7].astype(np.float32))
                rec[f"a{i}_vel"].append(st[7:13].astype(np.float32))
                rec[f"a{i}_setpoint_pos"].append(b["sp"].astype(np.float32))
                rec[f"a{i}_setpoint_yaw"].append(np.float32(b["yaw_sp"]))
                rec[f"a{i}_rotor_action"].append(cmd[0].numpy())
                rec[f"a{i}_arm_target"].append(b["arm_target"].astype(np.float32))
                rec[f"a{i}_arm_q"].append(b["q"].astype(np.float32))
                rec[f"a{i}_grasp_point"].append(grasp_point(b).astype(np.float32))
                rec[f"a{i}_integral_acc"].append(b["ctrl"].integral_acc[0].numpy().copy())
            rec["t"].append(np.float32(step * dt))
            rec["phase"].append(np.int8(pid))
            rec["door_open"].append(doors_now.copy())
            rec["door_held"].append(np.array(door_held))
            rec["button_pos"].append((button_now - origin).astype(np.float32))
            rec["button_held"].append(button_held)

            td = step_mdp(env.step(td))
            step += 1
            refresh()
            button_now = button_pose()[0]
            doors_now = cabinet.get_door_open_fraction()[0].cpu().numpy()

            if frames_dir is not None and step % args.video_every == 0:
                env.sim.render()
                capture_viewport_to_file(vp, str(frames_dir / f"{frame:05d}.png"))
                env.sim.render()
                frame += 1

            for i, b in enumerate(agents):
                if b["st"][2] < 0.2 or q_up_z(b["st"][3:7]) < 0.5:
                    status = f"agent{i}_crashed"
            if kind == "open":
                # The reason these are the fully-actuated airframe: it goes
                # round the arc sideways, level, instead of leaning into it.
                pull_tilt[dd] = max(pull_tilt[dd], math.degrees(
                    math.acos(max(-1.0, min(1.0, q_up_z(a["st"][3:7]))))))
            if status != "ok":
                break

        # ---- end of phase: grasp decisions, and a line for the log ------------
        if status == "ok" and kind == "close":
            gp, h = grasp_point(a), handle(dd, 0.0)
            grasp_err[dd] = float(np.linalg.norm(gp[:2] - h[:2]))
            gap = sum(a["q"][a["joint_names"].index(j)] for j in ("arm_lfinger_joint", "arm_rfinger_joint"))
            on_bar = grasp_err[dd] < GRASP_TOL_M and abs(gp[2] - h[2]) < bar_half_len - 0.02
            closed = abs(gap - 2.0 * bar_half) < 0.006
            print(f"[demo]   door {dd}: grasp point {grasp_err[dd] * 1e3:.1f} mm off the handle axis, "
                  f"{(gp[2] - h[2]) * 1e3:+.1f} mm along it; finger gap {gap * 1e3:.1f} mm on a "
                  f"{2e3 * bar_half:.0f} mm bar -> {'GRASPED' if on_bar and closed else 'missed'}", flush=True)
            if on_bar and closed:
                door_held[dd] = True
                a["door_frac"] = 0.0
        if status == "ok" and pname == "retriever_close":
            gp, c = grasp_point(a), cap_center()
            button_err = float(np.linalg.norm(gp[:2] - c[:2]))
            ok = button_err < BUTTON_TOL_M and abs(gp[2] - c[2]) < 0.015
            print(f"[demo]   button: tool axis {button_err * 1e3:.1f} mm off the button's, "
                  f"{(gp[2] - c[2]) * 1e3:+.1f} mm in height -> {'GRASPED' if ok else 'missed'}", flush=True)
            if ok:
                hp, hq = link_pose(a, "Housing")
                button_rel = []
                for view in button_views:
                    bp, bq = button_pose(view)
                    button_rel.append((q_rot(q_conj(hq), bp - hp), q_mul(q_conj(hq), bq)))
                    view.disable_gravities()
                button_held = True

        err = [float(np.linalg.norm(drone_pos(b) - b["sp"])) for b in agents]
        tilt = [math.degrees(math.acos(max(-1.0, min(1.0, q_up_z(b["st"][3:7]))))) for b in agents]
        print(f"[demo] {pname:18s} t={step * dt:6.2f}s  setpoint err (m) {np.round(err, 3).tolist()}  "
              f"tilt (deg) {np.round(tilt, 1).tolist()}  doors {np.round(doors_now, 3).tolist()}  "
              f"I-term (m/s^2) {[np.round(b['ctrl'].integral_acc[0].numpy(), 2).tolist() for b in agents]}",
              flush=True)
        if frames_dir is not None:
            env.sim.render()
            capture_viewport_to_file(vp, str(frames_dir.parent / f"{frames_dir.name}_{pid:02d}_{pname}.png"))
            env.sim.render()
        if status != "ok":
            break

    # ---- outcome ---------------------------------------------------------------
    doors = doors_now
    doors_open = bool((doors >= env.door_open_threshold).all())
    bpos = button_now - origin
    out_of_cabinet = float(np.dot(bpos[:2] - front_xy, front)) > 0.5
    success = status == "ok" and doors_open and (
        args.only == "doors" or (button_held and out_of_cabinet))
    print(f"[demo] status={status} doors_open={np.round(doors, 3).tolist()} "
          f"handle_grasp_err_mm={np.round(np.array(grasp_err) * 1e3, 1).tolist()} "
          f"worst_door_slip_mm={np.round(np.array(door_slip) * 1e3, 1).tolist()} "
          f"worst_tilt_while_pulling_deg={np.round(pull_tilt, 2).tolist()} "
          f"button_grasp_err_mm={button_err * 1e3:.1f} button_held={button_held} "
          f"button_final={np.round(bpos, 3).tolist()} out_of_cabinet={out_of_cabinet} "
          f"SUCCESS={success} ({step} steps, {step * dt:.1f} s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **{k: np.stack(v) if v else np.zeros(0) for k, v in rec.items()},
        success=np.bool_(success), status=np.str_(status), dt=np.float32(dt),
        phase_names=np.array([p for p, _ in PHASES]),
        worst_door_slip_m=np.array(door_slip, np.float32),
        worst_tilt_while_pulling_deg=np.array(pull_tilt, np.float32),
        presets=np.array([str(c.preset) for c in cfg.task.aerial_manipulators]),
        arm_dynamics_residual=np.array([
            a["am"].cfg.arm_dynamics_residual if args.arm_residual is None else args.arm_residual
            for a in agents], np.float32),
        pid_kp=np.stack([a["ctrl"].kp.numpy() for a in agents]),
        pid_ki=np.stack([a["ctrl"].ki.numpy() for a in agents]),
        pid_kd=np.stack([a["ctrl"].kd.numpy() for a in agents]),
        action_layout=np.str_("per agent: rotor commands in [-1, 1] from the PID position controller | "
                              "arm_target: joint position targets (rad, m for the hexa's fingers), teleop order"),
        grasp_model=np.str_("fingers close to the handle / button-cap width with gripper contacts filtered; "
                            "a held door's hinge follows the gripper's angle about it and its damping is "
                            "applied to the drone's base_link; the held button is carried kinematically"),
    )
    print(f"[demo] trajectory -> {out}", flush=True)

    if frames_dir is not None and frame:
        import imageio_ffmpeg  # its bundled ffmpeg; a system build may lack the png decoder

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-framerate", str(round(1 / (dt * args.video_every))),
                        "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.video], check=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[demo] video -> {args.video}", flush=True)
    exit_code = 0 if success else 1
except Exception:
    import traceback

    traceback.print_exc()
    exit_code = 2
# Kit ends the process inside close() with status 0, so the status has to be
# set this way rather than by raising after it.
if exit_code:
    os._exit(exit_code)
simulation_app.close()
