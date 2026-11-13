"""Scripted demonstration for AerialTransport: two iris650_soarm100 pick the
carry box up by one handle each and set it down inside the drop table's green
rectangle. Writes one ``.npz`` per successful episode (same container the
repo's other demo data uses, ``aetos_multiam/data/*/episodes/ep###.npz``);
failed attempts go to ``<out-dir>/failed/`` for diagnosis only.

    export ISAACSIM_PATH=/home/benhua/isaacsim
    conda activate omnidrone
    python scripts/collect_aerial_transport_demo.py --num-episodes 10 --video-dir outputs/aerial_transport

Per episode (``--seed`` makes it reproducible): each drone spawns uniformly
within +-SPAWN_XY_M of its painted zone centre, at SPAWN_Z_RANGE, heading at
the pick table (the arm reaches out along the nose) +-SPAWN_YAW_DEG, so
neither robot has to turn around before the approach; the box is placed +-BOX_XY_M on the pick table with yaw
+-BOX_YAW_DEG. Approach directions follow the box's yaw, and the carry
rotates the box back square so its footprint lands inside the target.

Physics setup -- the "isolated" plant the follow-up whole-body controller is
meant to drop:

* arm dynamics frozen (``freeze_arm_dynamics``, the preset's residual 0.1), and
  ``shoulder_pan`` held at 0 (preset's ``arm_locked_joints``) -- yaw comes from
  the drone instead; the other arm joints are PD-driven.
* Each drone is flown on a scripted setpoint through the PX4 offboard
  interface (``--drone-action ctbr``, default): the Lee position loop plus a
  PX4-style attitude P loop give body rates + collective thrust (the stored
  action), and a PX4-style rate PID closes them every 4 ms physics substep
  (``controllers/drone/px4_rate_controller.py``). ``--drone-action rotor``
  is the old Lee -> per-rotor path.
* The handle bar enters the open jaw gap and the jaw closes until it touches
  the bar (geometric contact against the real collision meshes -- the
  box/arm contacts themselves are filtered, see below). Every step, both
  fingers' collision-mesh vertices are checked against each handle-bar
  cylinder; an episode whose worst penetration exceeds ``PEN_FAIL_MM`` fails
  and is not written. Once both jaws are closed within ``GRASP_TOL_M`` of
  their bar, the box is carried
  KINEMATICALLY, gravity off: each gripper link carries its bar centre at the
  link-frame point recorded at grasp time; the box is fitted through those
  two points (long axis along the line joining them, roll from the grippers'
  average orientation). A height or sideways mismatch between the robots then
  just tilts the box, and a spacing mismatch slides the bar along the jaw
  slot's depth, where there is room -- averaging two rigid poses instead
  pushed the bar 9 mm into a finger. Each drone then
  gets half the box weight as an external force on base_link, and its
  controller's mass feedforward is raised by the same half -- lumped at the
  CoM exactly like ``freeze_arm_dynamics`` lumps the arm. Applied at the grasp
  point instead, the 1.7 N.m lever-arm moment makes each (integral-free) Lee
  controller lean in and creep 0.24 m toward the box, i.e. the two grasp
  points close 0.48 m on a rigid box. Box <-> arm-link collisions are
  filtered out (authored before the sim starts), so the jaw closes to its
  on-bar angle instead of the carried box shoving the near-massless frozen
  links around -- which, measured, bent the elbow 1 rad and dragged both
  drones 0.15 m off their setpoints. A friction carry is not used:
  with the arm frozen the jaw link weighs ~1 g against a 1 kg box, which the
  solver does not hold. shortcut: closed-chain payload dynamics (box inertia,
  internal forces between the two robots) are not simulated; upgrade to a
  physical attachment (fixed joints) once a whole-body controller exists.
"""
import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PHYSICS_DT, SUBSTEPS = 0.004, 4
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out-dir", default=str(REPO / "aetos_multiam/data/aerial_transport/episodes"))
parser.add_argument("--num-episodes", type=int, default=10, help="Attempts; only successes are written.")
parser.add_argument("--start-index", type=int, default=0, help="File index of the first written episode.")
parser.add_argument("--video-dir", default=None, help="Optional: write ep###.mp4 per episode (overview camera).")
parser.add_argument("--video-every", type=int, default=4, help="Capture one frame every N control steps.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--drone-action", choices=["ctbr", "rotor"], default="ctbr",
                    help="ctbr: expert emits PX4 body rates + thrust, closed by a PX4-style rate loop at "
                         "1/PHYSICS_DT (default). rotor: the old Lee -> per-rotor command path, for comparison.")
parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim window instead of headless.")
args = parser.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=[
        "task=AerialTransport", f"headless={str(not args.gui).lower()}",
        "++task.env.num_envs=1", "++task.env.max_episode_length=4000",
        # 62.5 Hz policy/expert, 250 Hz physics + rate loop.
        "++task.filter_box_arm_contacts=true",
        f"++task.sim.dt={PHYSICS_DT}", f"++task.sim.substeps={SUBSTEPS}", f"++task.drone_action={args.drone_action}",
    ])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
simulation_app = init_simulation_app(cfg)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf  # noqa: E402
from torchrl.envs.utils import step_mdp  # noqa: E402

from omni_drones.utils.torch import (  # noqa: E402
    euler_to_quaternion, quat_axis, quat_mul, quat_rotate, quat_rotate_inverse, quaternion_to_euler,
)


def rotmat_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation -> wxyz."""
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return np.array([w, x, y, z])


def quat_conj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)
from aetos_multiam.controllers import ctbr_from_lee, make_position_controller, normalize_ctbr  # noqa: E402
from aetos_multiam.environments import make as make_env  # noqa: E402
from aetos_multiam.environments.collaborative.aerial_transport import (  # noqa: E402
    DROP_TABLE_Y_M, PICK_TABLE_Y_M, TABLE_SIZE_M, TABLE_TOP_Z_M, TARGET_MARGIN_M,
)
from aetos_multiam.objects import CarryBox  # noqa: E402
from aetos_multiam.utils import freeze_arm_dynamics  # noqa: E402

G = 9.81
#: Where the handle-bar axis sits when grasped, in the GRIPPER LINK frame
#: (link -z points out along the fingers, +x across the gap towards the jaw).
#: Measured from the SO101 collision meshes with the jaw open at 1.5 rad: the
#: bar (r = 15 mm) can slide in clear of both fingers for h in [11.6, 79.6] mm
#: (h = distance across the gap from the fixed finger side) down to 51 mm
#: deep; this is 68 mm deep (17 mm short of the palm, room for overshoot),
#: h = 12 mm (7 mm clear of the fixed finger; the jaw closes onto the bar
#: from the other side).
#: (The first version guessed this from bounding boxes in the drone frame and
#: put the bar 12 mm inside the fixed finger.)
GRASP_LOCAL = (0.0108, -0.0004, -0.0682)
GRIP_OPEN, GRIP_CLOSED = 1.5, 0.0
#: The jaw stops closing once its surface is this close to the bar.
CONTACT_STOP_M = 0.0015  # the drive coasts ~1-2 mm past the stop
#: Worst bar penetration (any finger, any step) an episode may have.
PEN_FAIL_MM = 1.0
PREGRASP_BACKOFF_M = 0.22
GRASP_TOL_M = 0.025
LIFT_M = 0.25
RETREAT_M = 0.30
#: Low-pass on the measured base_link -> grasp-point offset (per step). An
#: integrator on grasp error was used first: it wound up on the drone's
#: tracking lag during the approach and drove the fingers 6 cm through the bar.
GFB_ALPHA = 0.05
#: The approach stops this far short; pre_close creeps the rest with a small
#: integrator on grasp-point error (only while the setpoint is slow, so it
#: cannot wind up on tracking lag the way a full-approach integrator did).
APPROACH_STOP_M, CREEP_GAIN, CREEP_MAX_M = 0.015, 0.8, 0.02
#: Set-down: ``lower`` brings the box to SETDOWN_FAST_M above the table, then
#: ``touchdown`` creeps at SETDOWN_SPEED_MPS until it is SETDOWN_GAP_M above
#: it. (Stopping on contact from the fast descent overshot 17 mm.)
SETDOWN_FAST_M, SETDOWN_SPEED_MPS, SETDOWN_GAP_M = 0.03, 0.01, 0.0005
#: The box is let go the moment it is SETDOWN_GAP_M above the table (it drops
#: that far onto it); holding it on longer let the drones sag and the table
#: clamp pressed the bar into the fixed finger. Before backing out, the
#: grippers lift UNHOOK_M off the bar, so the pitch of the retreat cannot drag
#: the fixed finger across it.
UNHOOK_M = 0.006
#: The table, not the robots, carries the box while it rests on it: load and
#: mass feedforward ramp in over the first LOAD_RAMP_M of lift-off (and out
#: again on set-down). Switching the full load on at grasp time dipped the
#: drones 5 mm and pressed the bar into the fixed finger against the table.
LOAD_RAMP_M = 0.01
#: Jaw (SO101 ``gripper`` DOF) drive gains. As authored, the drive cannot hold
#: the frozen ~1 g link: once stopped on the bar it sagged 4 mm off it, and
#: swung +-7 mm under transport accelerations. Servoing its target chased the
#: swing into the bar (6 mm); writing the joint position directly knocked the
#: whole arm 10 cm. Authored: kp 0.35 N.m/rad, kd 0.049 -- 1/20 of the wrist
#: joints (6.9-48). Raised to wrist_roll's order; a real servo gripper holds.
JAW_KP, JAW_KD = 10.0, 0.5
SPAWN_XY_M, SPAWN_Z_RANGE, SPAWN_YAW_DEG = 0.6, (1.0, 1.5), 10.0
BOX_XY_M, BOX_YAW_DEG = 0.10, 15.0

PHASES = [  # (name, seconds)
    ("settle", 1.0), ("fly_to_pregrasp", 7.0), ("align", 2.0), ("approach", 5.0),
    ("pre_close", 2.5), ("close", 2.5), ("lift", 2.5), ("transport", 6.0),
    ("lower", 3.0), ("touchdown", 3.5), ("open", 1.0), ("unhook", 0.8), ("retreat", 2.5), ("ascend", 2.0), ("hold", 1.0),
]


def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def yaw_rot(yaw: float, v) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


from omni.isaac.core.utils.stage import get_current_stage  # noqa: E402

try:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    env = make_env("AerialTransport", cfg, headless=not args.gui)
    env.reset()
    dev = env.device
    dt = float(env.sim.get_physics_dt()) * env.substeps
    ams = env.aerial_manipulators
    box_view = env.carry_box._view
    keys = env._agent_keys
    env_origin = env.envs_positions[0].cpu().numpy()

    agents = []
    for i, am in enumerate(ams):
        d = am.drone
        # Once per process: freeze_arm_dynamics is not idempotent.
        freeze_arm_dynamics(am, residual=float(cfg.task.get("arm_dynamics_residual", am.cfg.arm_dynamics_residual)))
        ctrl = make_position_controller(d, G).to(dev)
        names = list(d._view._body_names)
        a = {
            "am": am, "drone": d, "ctrl": ctrl,
            "side": -1.0 if i == 0 else 1.0,  # agent 0 owns the -x handle
            "zone": np.array(env._default_spawn_translation(i), dtype=np.float64),
            "gripper_link": names.index("gripper"), "n_links": len(names),
            "load": np.zeros(3), "mass0": float(ctrl.mass.data),
        }

        # Half the box weight on base_link: wrap the one base_link force write
        # apply_action makes, so the drag/torques it computes stay untouched
        # (a second apply call would replace them, not add).
        orig_apply = d.base_link.apply_forces_and_torques_at_pos

        def patched(forces=None, torques=None, positions=None, is_global=True, _a=a, _orig=orig_apply):
            if np.any(_a["load"]) and forces is not None:
                forces = forces + torch.as_tensor(_a["load"], device=dev, dtype=forces.dtype).reshape(1, 3)
            return _orig(forces, torques, positions, is_global=is_global)

        d.base_link.apply_forces_and_torques_at_pos = patched
        agents.append(a)

    def drone_yaw(a) -> float:
        return float(quaternion_to_euler(a["drone"].get_state()[0, 0, 3:7].reshape(1, 4))[0, 2])

    def link_pose(a, name):
        """(pos, quat wxyz) of one link, world frame. PhysX returns xyzw."""
        lt = a["drone"]._view._physics_view.get_link_transforms().reshape(-1, a["n_links"], 7)[0][a["names"].index(name)]
        return lt[:3], torch.cat([lt[6:7], lt[3:6]]).reshape(1, 4)

    grasp_local = torch.tensor([GRASP_LOCAL], device=dev)

    def grasp_point(a) -> np.ndarray:
        p, q = link_pose(a, "gripper")
        return (p + quat_rotate(q, grasp_local)[0]).cpu().numpy() - env_origin

    # Collision-mesh vertices of both fingers, in their own link frames.
    from pxr import Usd, UsdGeom

    def collision_points(link_path):
        link = get_current_stage().GetPrimAtPath(link_path)
        to_link = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        pts = []
        for prim in Usd.PrimRange(link, Usd.PrimAllPrimsPredicate):
            if prim.IsA(UsdGeom.Mesh) and "collision" in prim.GetPath().pathString.lower():
                m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * to_link
                pts += [list(m.Transform(v)) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
        if not pts:
            raise RuntimeError(f"no collision mesh under {link_path}")
        return torch.tensor(pts[::4], device=dev, dtype=torch.float32)

    for i, a in enumerate(agents):
        jaw_idx = a["am"].teleop_joint_indices[5:6]
        a["drone"]._view.set_gains(kps=torch.tensor([[JAW_KP]], device=dev), kds=torch.tensor([[JAW_KD]], device=dev),
                                   joint_indices=jaw_idx)
        kp_now = float(a["drone"]._view.get_gains()[0][0, int(jaw_idx[0])])
        if abs(kp_now - JAW_KP) > 1e-3:
            raise RuntimeError(f"jaw gain write did not land: kp reads {kp_now}")
        a["names"] = list(a["drone"]._view._body_names)
        a["finger_pts"] = {ln: collision_points(f"{env.template_env_ns}/am_{i}_0/manipulator/{ln}")
                           for ln in ("gripper", "jaw")}

    bar_r = CarryBox.handle_bar_diameter_m / 2

    def bar_clearance(i) -> dict:
        """Smallest distance (mm) from each finger's surface to handle bar i;
        negative = penetration. Only the bar's own length counts."""
        a = agents[i]
        bp, bq = box_view.get_world_poses()
        hx, hy, hz = env.carry_box.handle_local_offsets()[i]
        out = {}
        for ln, pts in a["finger_pts"].items():
            p, q = link_pose(a, ln)
            w = quat_rotate(q.expand(len(pts), 4), pts) + p
            loc = quat_rotate_inverse(bq.reshape(1, 4).expand(len(pts), 4), w - bp.reshape(1, 3))
            on_bar = (loc[:, 1] - hy).abs() < CarryBox.handle_bar_length_m / 2
            r = torch.hypot(loc[:, 0] - hx, loc[:, 2] - hz)[on_bar]
            out[ln] = 1000.0 * (float(r.min()) - bar_r) if len(r) else 1e3
        return out

    def box_pose():
        bp, bq = box_view.get_world_poses()
        return bp[0].cpu().numpy() - env_origin, bq[0].reshape(1, 4)

    def box_yaw() -> float:
        return float(quaternion_to_euler(box_pose()[1])[0, 2])

    def live_handle(i: int) -> np.ndarray:
        # The box can move; aim at where the bar IS.
        bp, bq = box_pose()
        off = torch.tensor(env.carry_box.handle_local_offsets()[i], device=dev, dtype=bq.dtype).reshape(1, 3)
        return bp + quat_rotate(bq, off)[0].cpu().numpy()

    def outward(a) -> np.ndarray:
        """Unit vector from the box out past this agent's handle."""
        return a["side"] * yaw_rot(box_yaw(), (1.0, 0.0, 0.0))

    def nominal_drone_pos(a, grasp_target: np.ndarray, yaw: float) -> np.ndarray:
        return grasp_target - yaw_rot(yaw, a["g_from_base"])

    # Rest pose with the arm reaching forward (all zeros).
    arm_q = torch.zeros(1, 6, device=dev)

    vp = None
    if args.video_dir:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        vp = get_active_viewport()
        cs = ViewportCameraState("/OmniverseKit_Persp", vp)
        cs.set_position_world(Gf.Vec3d(2.5, -5.5, 3.2), True)
        cs.set_target_world(Gf.Vec3d(0.0, 1.0, 0.7), True)
        for _ in range(20):
            env.sim.render()  # render() pauses physics; simulation_app.update() would step it

    def run_episode(video_path):
        # --- randomized reset -------------------------------------------------
        td = env.reset()
        init = {}
        for i, a in enumerate(agents):
            pos = a["zone"] + np.array([*rng.uniform(-SPAWN_XY_M, SPAWN_XY_M, 2), 0.0])
            pos[2] = rng.uniform(*SPAWN_Z_RANGE)
            to_table = np.array(env.pick_table_center[:2]) - pos[:2]
            yaw = math.atan2(to_table[1], to_table[0]) + math.radians(rng.uniform(-SPAWN_YAW_DEG, SPAWN_YAW_DEG))
            a["drone"].set_world_poses(
                torch.as_tensor(pos + env_origin, device=dev, dtype=torch.float32).reshape(1, 3),
                euler_to_quaternion(torch.tensor([[0.0, 0.0, yaw]], device=dev)))
            a["drone"].set_velocities(torch.zeros(1, 6, device=dev))
            # Rotors still spinning from the last episode turn the teleport into
            # a 7-11 rad/s body spin (measured); start every episode at rest.
            ndof = a["drone"]._view.num_dof
            a["drone"]._view.set_joint_velocities(torch.zeros(1, 1, ndof, device=dev),
                                                  joint_indices=torch.arange(ndof, device=dev))
            a["drone"].throttle.zero_()
            a.update(spawn=pos, spawn_yaw=yaw, creep=np.zeros(3), clear={"gripper": 1e3, "jaw": 1e3})
            a["load"][:] = 0.0
            a["ctrl"].mass.data.fill_(a["mass0"])
            for k in ("sp", "yaw_sp", "grip", "sp_vel", "touched", "gfb"):
                a.pop(k, None)
            init[f"a{i}_spawn_pos"], init[f"a{i}_spawn_yaw"] = pos.astype(np.float32), np.float32(yaw)
        bxy = rng.uniform(-BOX_XY_M, BOX_XY_M, 2)
        byaw = math.radians(rng.uniform(-BOX_YAW_DEG, BOX_YAW_DEG))
        bpos = np.array([bxy[0], PICK_TABLE_Y_M + bxy[1], TABLE_TOP_Z_M + 0.002])
        box_view.enable_gravities()
        box_view.set_world_poses(torch.as_tensor(bpos + env_origin, device=dev, dtype=torch.float32).reshape(1, 3),
                                 euler_to_quaternion(torch.tensor([[0.0, 0.0, byaw]], device=dev)))
        box_view.set_velocities(torch.zeros(1, 6, device=dev))
        init["box_init_pos"], init["box_init_yaw"] = bpos.astype(np.float32), np.float32(byaw)
        print(f"[demo] reset: spawns={[np.round(a['spawn'], 2).tolist() for a in agents]} "
              f"box_xy={np.round(bxy, 3).tolist()} box_yaw={math.degrees(byaw):.1f}deg", flush=True)

        frames_dir = None
        if video_path is not None:
            frames_dir = Path(video_path).with_suffix("")
            shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True)

        rec = {k: [] for k in ["t", "phase", "attached", "box_pos", "box_quat", "box_vel"]}
        for i in range(len(agents)):
            for k in ["obs", "action", "drone_action", "rate_sp", "thrust_frac", "body_rate", "arm_target", "drone_pos", "drone_quat", "drone_vel",
                      "setpoint_pos", "setpoint_vel", "setpoint_yaw", "arm_q", "arm_dq", "grasp_point",
                      "clear_gripper_mm", "clear_jaw_mm"]:
                rec[f"a{i}_{k}"] = []

        attached = False
        attach_rel = prev_bp = None
        lowered = False
        status = "ok"
        step = frame = 0
        worst_pen = 0.0

        for pid, (pname, dur) in enumerate(PHASES):
            n = max(1, int(round(dur / dt)))
            # Snapshot what each phase interpolates from.
            snap = [{"sp": a.get("sp", a["spawn"]).copy(), "yaw": a.get("yaw_sp", a["spawn_yaw"]),
                     "grip": a.get("grip", GRIP_CLOSED)} for a in agents]
            box_p0 = box_pose()[0]
            box_yaw0 = box_yaw()
            if pname == "fly_to_pregrasp":
                for a in agents:
                    # Grasp point relative to base_link, heading frame, as the
                    # arm actually hangs (nominal feedforward; the alignment
                    # integrator takes the rest).
                    st = a["drone"].get_state()[0, 0]
                    a["g_from_base"] = yaw_rot(-drone_yaw(a), grasp_point(a) - st[:3].cpu().numpy())
                    # Face the box along its own axis; unwrap to the nearest turn.
                    goal = math.atan2(-outward(a)[1], -outward(a)[0])
                    a["yaw_goal"] = a["spawn_yaw"] + wrap(goal - a["spawn_yaw"])
            if pname == "retreat":
                for a in agents:
                    a["out"] = outward(a)

            if pname == "lift" and not attached:
                status = "grasp_failed"
                break
            if pname == "open" and attached:  # touchdown never saw the gap close; let go anyway
                attached = False
                box_view.enable_gravities()
                for a in agents:
                    a["load"][:] = 0.0
                    a["ctrl"].mass.data.fill_(a["mass0"])

            for k in range(n):
                u = min_jerk((k + 1) / n)
                for i, a in enumerate(agents):
                    s = snap[i]
                    yaw_sp, grip = s["yaw"], s["grip"]
                    if pname == "settle":
                        sp = a["spawn"].copy()
                    elif pname == "fly_to_pregrasp":
                        goal = nominal_drone_pos(a, live_handle(i) + outward(a) * PREGRASP_BACKOFF_M, a["yaw_goal"])
                        sp = s["sp"] + u * (goal - s["sp"])
                        yaw_sp = a["spawn_yaw"] + u * (a["yaw_goal"] - a["spawn_yaw"])
                        grip = s["grip"] + u * (GRIP_OPEN - s["grip"])
                    elif pname in ("align", "approach", "pre_close"):
                        if pname == "align":
                            back = PREGRASP_BACKOFF_M
                        elif pname == "approach":
                            back = APPROACH_STOP_M + (PREGRASP_BACKOFF_M - APPROACH_STOP_M) * (1.0 - u)
                        else:
                            back = APPROACH_STOP_M * (1.0 - u)
                        gt = live_handle(i) + outward(a) * back
                        gp = grasp_point(a)
                        gfb = gp - a["drone"].get_state()[0, 0, :3].cpu().numpy()
                        a["gfb"] = gfb if "gfb" not in a else (1 - GFB_ALPHA) * a["gfb"] + GFB_ALPHA * gfb
                        if pname == "pre_close":  # integrating earlier winds up on transients
                            a["creep"] = np.clip(a["creep"] + CREEP_GAIN * (gt - gp) * dt, -CREEP_MAX_M, CREEP_MAX_M)
                        sp = gt - a["gfb"] + a["creep"]
                        yaw_sp = a["yaw_goal"]
                    elif pname == "close":
                        sp = s["sp"]
                        # Close until the jaw touches the bar, then hold there.
                        grip = a.get("touched", s["grip"] + u * (GRIP_CLOSED - s["grip"]))
                    elif pname == "lift":
                        sp = s["sp"] + u * np.array([0, 0, LIFT_M])
                        grip = a["touched"]
                    elif pname == "transport":
                        # Carry the box centre to the target and turn it square,
                        # both drones rigidly about that centre.
                        c0 = np.array([box_p0[0], box_p0[1], 0.0])
                        c = c0 + u * (np.array([0.0, DROP_TABLE_Y_M, 0.0]) - c0)
                        dpsi = -u * box_yaw0
                        sp = c + yaw_rot(dpsi, s["sp"] - c0)
                        yaw_sp = s["yaw"] + dpsi
                        grip = a["touched"]
                    elif pname == "lower":
                        sp = s["sp"] + u * np.array([0, 0, TABLE_TOP_Z_M + SETDOWN_FAST_M - box_p0[2]])
                        grip = a["touched"]
                    elif pname == "touchdown":
                        # Creep down until the measured box gap is SETDOWN_GAP_M.
                        sp = a["sp"] if lowered else a["sp"] - np.array([0, 0, SETDOWN_SPEED_MPS * dt])
                        grip = a["touched"]
                    elif pname == "open":
                        sp = s["sp"]
                        grip = s["grip"] + u * (GRIP_OPEN - s["grip"])
                    elif pname == "unhook":
                        sp = s["sp"] + u * np.array([0, 0, UNHOOK_M])
                    elif pname == "retreat":
                        sp = s["sp"] + u * a["out"] * RETREAT_M
                    elif pname == "ascend":
                        sp = s["sp"] + u * np.array([0, 0, 0.5])
                    else:
                        sp = s["sp"]
                    prev = a.get("sp", sp)
                    a["sp_vel"] = (sp - prev) / dt
                    a["sp"], a["yaw_sp"], a["grip"] = sp, yaw_sp, grip

                # --- grasp bookkeeping ----------------------------------------
                gps = [grasp_point(a) for a in agents]
                clear = [bar_clearance(j) for j in range(len(agents))]
                worst_pen = max([worst_pen] + [-c for cl in clear for c in cl.values()])
                for a, cl in zip(agents, clear):
                    a["clear"] = cl
                if pname == "close":
                    for a, cl in zip(agents, clear):
                        if "touched" not in a and cl["jaw"] <= 1000.0 * CONTACT_STOP_M:
                            # Freeze at where the jaw actually IS, not the command.
                            a["touched"] = float(a["drone"]._view.get_joint_positions()[0, 0][a["am"].teleop_joint_indices[5]])
                if pname == "close" and k == n - 1:
                    errs = [float(np.linalg.norm(gp - live_handle(j))) for j, gp in enumerate(gps)]
                    print(f"[demo] grasp errors at close: {[round(e, 4) for e in errs]} m, finger/jaw clearance "
                          f"{[(round(c['gripper'], 2), round(c['jaw'], 2)) for c in clear]} mm", flush=True)
                    if max(errs) < GRASP_TOL_M:
                        attached = True
                        # Bar centre and box orientation in each gripper link's
                        # frame, held from now on.
                        wbq = box_view.get_world_poses()[1].reshape(1, 4)
                        attach_rel = []
                        for j, a in enumerate(agents):
                            p, q = link_pose(a, "gripper")
                            hc = torch.as_tensor(live_handle(j) + env_origin, device=dev, dtype=torch.float32)
                            attach_rel.append((quat_rotate_inverse(q, (hc - p).reshape(1, 3)),
                                               quat_mul(quat_conj(q), wbq)))
                        prev_bp = box_pose()[0]
                        box_view.disable_gravities()

                if attached:
                    # Box through the two carried bar centres (see docstring).
                    cs, ys = [], []
                    for a, (r_rel, q_rel) in zip(agents, attach_rel):
                        p, q = link_pose(a, "gripper")
                        cs.append((p + quat_rotate(q, r_rel)[0]).cpu().numpy())
                        ys.append(quat_rotate(quat_mul(q, q_rel), torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0].cpu().numpy())
                    bx = (cs[1] - cs[0]) / np.linalg.norm(cs[1] - cs[0])
                    by = ys[0] + ys[1]
                    by = by - bx * (bx @ by)
                    by /= np.linalg.norm(by)
                    rot = np.stack([bx, by, np.cross(bx, by)], axis=1)
                    mid_local = np.mean(env.carry_box.handle_local_offsets(), axis=0)
                    bp = (cs[0] + cs[1]) / 2 - rot @ mid_local - env_origin
                    bq = torch.as_tensor(rotmat_to_quat(rot), device=dev, dtype=torch.float32).reshape(1, 4)
                    # A table under the box holds it up (the kinematic box would
                    # otherwise sink into it) and carries its weight near it.
                    gap = np.inf
                    for ty in (PICK_TABLE_Y_M, DROP_TABLE_Y_M):
                        if abs(bp[0]) < TABLE_SIZE_M[0] / 2 and abs(bp[1] - ty) < TABLE_SIZE_M[1] / 2:
                            gap = bp[2] - TABLE_TOP_Z_M
                            bp[2] = max(bp[2], TABLE_TOP_Z_M)
                    load_frac = float(np.clip(gap / LOAD_RAMP_M, 0.0, 1.0))
                    vel = torch.zeros(1, 6, device=dev)
                    vel[0, :3] = torch.as_tensor((bp - prev_bp) / dt, device=dev)
                    prev_bp = bp
                    box_view.set_world_poses(torch.as_tensor(bp + env_origin, device=dev,
                                                             dtype=torch.float32).reshape(1, 3), bq)
                    box_view.set_velocities(vel)
                    for a in agents:
                        a["load"][:] = (0.0, 0.0, -load_frac * CarryBox.mass_kg * G / len(agents))
                        a["ctrl"].mass.data.fill_(a["mass0"] + load_frac * CarryBox.mass_kg / len(agents))
                    if pname == "touchdown" and not lowered and gap <= SETDOWN_GAP_M:
                        lowered = True
                        attached = False
                        box_view.enable_gravities()
                        for a in agents:
                            a["load"][:] = 0.0
                            a["ctrl"].mass.data.fill_(a["mass0"])

                # --- commands -------------------------------------------------
                for i, (key, a) in enumerate(zip(keys, agents)):
                    d = a["drone"]
                    st = d.get_state()[..., :13].squeeze(0)
                    tgt_p = torch.as_tensor(a["sp"], device=dev, dtype=torch.float32).reshape(1, 3)
                    tgt_v = torch.as_tensor(a["sp_vel"], device=dev, dtype=torch.float32).reshape(1, 3)
                    tgt_yaw = torch.tensor([[a["yaw_sp"]]], device=dev)
                    if args.drone_action == "ctbr":
                        rate_sp, thrust_frac = ctbr_from_lee(a["ctrl"], st, tgt_p, tgt_v, tgt_yaw,
                                                             env.rate_controllers[i].max_total_thrust)
                        rotor = normalize_ctbr(rate_sp, thrust_frac)  # the drone slice of the action
                    else:
                        rotor = a["ctrl"].compute(st, target_pos=tgt_p, target_vel=tgt_v, target_yaw=tgt_yaw.reshape(1))
                        rate_sp, thrust_frac = torch.full((1, 3), float("nan")), torch.full((1, 1), float("nan"))
                    q = arm_q.clone()
                    q[0, 5] = a["grip"]
                    a["am"].hold_joints(q)

                    action = torch.cat([rotor.reshape(1, -1), q], dim=-1)
                    td.set((key, "action"), action.reshape(1, 1, -1))

                    full = d.get_state()[0, 0]
                    v = d._view
                    rec[f"a{i}_obs"].append(td[key, "observation"].reshape(-1).cpu().numpy())
                    rec[f"a{i}_action"].append(action.reshape(-1).cpu().numpy())
                    rec[f"a{i}_drone_action"].append(rotor.reshape(-1).cpu().numpy())
                    rec[f"a{i}_rate_sp"].append(rate_sp.reshape(-1).cpu().numpy())
                    rec[f"a{i}_thrust_frac"].append(np.float32(float(thrust_frac.reshape(-1)[0])))
                    rec[f"a{i}_body_rate"].append(quat_rotate_inverse(st[:, 3:7], st[:, 10:13]).reshape(-1).cpu().numpy())
                    rec[f"a{i}_arm_target"].append(q.reshape(-1).cpu().numpy())
                    rec[f"a{i}_drone_pos"].append(full[:3].cpu().numpy())
                    rec[f"a{i}_drone_quat"].append(full[3:7].cpu().numpy())
                    rec[f"a{i}_drone_vel"].append(full[7:13].cpu().numpy())
                    rec[f"a{i}_setpoint_pos"].append(a["sp"].astype(np.float32))
                    rec[f"a{i}_setpoint_vel"].append(a["sp_vel"].astype(np.float32))
                    rec[f"a{i}_setpoint_yaw"].append(np.float32(a["yaw_sp"]))
                    rec[f"a{i}_arm_q"].append(v.get_joint_positions()[0, 0][a["am"].teleop_joint_indices].cpu().numpy())
                    rec[f"a{i}_arm_dq"].append(v.get_joint_velocities()[0, 0][a["am"].teleop_joint_indices].cpu().numpy())
                    rec[f"a{i}_grasp_point"].append(gps[i].astype(np.float32))
                    rec[f"a{i}_clear_gripper_mm"].append(np.float32(clear[i]["gripper"]))
                    rec[f"a{i}_clear_jaw_mm"].append(np.float32(clear[i]["jaw"]))

                bp_t, bq_t = box_pose()
                rec["box_pos"].append(bp_t.astype(np.float32))
                rec["box_quat"].append(bq_t[0].cpu().numpy())
                rec["box_vel"].append(box_view.get_velocities()[0].cpu().numpy())
                rec["t"].append(np.float32(step * dt))
                rec["phase"].append(np.int8(pid))
                rec["attached"].append(attached)

                td = step_mdp(env.step(td))
                step += 1

                if frames_dir is not None and step % args.video_every == 0:
                    env.sim.render()
                    capture_viewport_to_file(vp, str(frames_dir / f"{frame:05d}.png"))
                    env.sim.render()
                    frame += 1

                for i, a in enumerate(agents):
                    st = a["drone"].get_state()[0, 0]
                    if float(st[2]) < 0.2 or float(quat_axis(st[3:7].reshape(1, 4), axis=2)[0, 2]) < 0.5:
                        status = f"agent{i}_crashed"
                if status != "ok":
                    break
            if status != "ok":
                break

        # --- outcome: whole footprint inside the green rectangle, resting flat ---
        bp, bq = box_pose()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(quat_axis(bq, axis=2)[0, 2])))))
        hx = CarryBox.body_size_m[0] / 2 + TARGET_MARGIN_M
        hy = CarryBox.body_size_m[1] / 2 + TARGET_MARGIN_M
        corners = [bp + yaw_rot(box_yaw(), (sx * CarryBox.body_size_m[0] / 2, sy * CarryBox.body_size_m[1] / 2, 0))
                   for sx in (-1, 1) for sy in (-1, 1)]
        in_rect = all(abs(c[0]) <= hx and abs(c[1] - DROP_TABLE_Y_M) <= hy for c in corners)
        on_table = abs(bp[2] - TABLE_TOP_Z_M) < 0.02
        if status == "ok" and worst_pen > PEN_FAIL_MM:
            status = "finger_penetrated_bar"
        success = status == "ok" and in_rect and on_table and tilt < 5.0
        print(f"[demo] status={status} box_final={np.round(bp, 4).tolist()} yaw={math.degrees(box_yaw()):.2f}deg "
              f"tilt={tilt:.2f}deg in_rect={in_rect} on_table={on_table} worst_pen={worst_pen:.2f}mm "
              f"jaw_on_bar={[round(a.get('touched', float('nan')), 3) for a in agents]} SUCCESS={success} ({step} steps)",
              flush=True)

        arrays = {k: np.stack(v) if v else np.zeros(0) for k, v in rec.items()}
        meta = dict(
            success=np.bool_(success), status=np.str_(status), dt=np.float32(dt), seed=np.int64(args.seed),
            worst_penetration_mm=np.float32(worst_pen),
            phase_names=np.array([p for p, _ in PHASES]),
            handle_pos=np.stack([live_handle(i) for i in range(len(agents))]).astype(np.float32),
            target_center=np.array([0.0, DROP_TABLE_Y_M, TABLE_TOP_Z_M], np.float32),
            target_half_extent=np.array([hx, hy], np.float32),
            presets=np.array([str(am.cfg.drone_type) + "+" + str(am.cfg.manipulator_type) for am in ams]),
            drone_action_mode=np.str_(args.drone_action),
            physics_dt=np.float32(PHYSICS_DT),
            action_layout=np.str_(
                ("per agent: [body rates p,q,r / (220,220,200 deg/s), 2*thrust_frac-1] in [-1,1] (PX4 offboard "
                 "CTBR; thrust_frac = collective / max total thrust; closed by PX4RateController at 1/physics_dt)"
                 if args.drone_action == "ctbr" else
                 "per agent: [rotor_cmd(4) in [-1,1] from LeePositionController]")
                + " | arm joint position targets (6, rad): shoulder_pan..wrist_roll, gripper"),
            grasp_model=np.str_("jaw closes on handle; box then carried kinematically from the two grasp "
                                "points, half its weight applied on each drone's base_link"),
            **init,
        )
        return success, {**arrays, **meta}, frames_dir, frame

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, failed = [], []
    idx = args.start_index
    for attempt in range(args.num_episodes):
        video = str(Path(args.video_dir) / f"ep{idx:03d}.mp4") if args.video_dir else None
        print(f"[demo] === attempt {attempt} -> ep{idx:03d} ===", flush=True)
        ok, data, frames_dir, nframes = run_episode(video)
        if not ok:
            failed.append((attempt, str(data["status"])))
            (out_dir / "failed").mkdir(exist_ok=True)  # for diagnosis, kept out of the dataset
            np.savez_compressed(out_dir / "failed" / f"attempt{attempt:03d}.npz", **data)
            if frames_dir is not None:
                shutil.rmtree(frames_dir, ignore_errors=True)
            continue
        path = out_dir / f"ep{idx:03d}.npz"
        np.savez_compressed(path, **data)
        written.append(path.name)
        if frames_dir is not None and nframes:
            import imageio_ffmpeg  # its bundled ffmpeg; a system build may lack the png decoder

            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                            "-framerate", str(round(1 / (dt * args.video_every))),
                            "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", video], check=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
        idx += 1
    print(f"[demo] SUMMARY: {len(written)}/{args.num_episodes} succeeded -> {out_dir} {written}; "
          f"failed attempts: {failed}", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
