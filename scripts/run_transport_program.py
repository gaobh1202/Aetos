"""Execute a planner's program (scripts/plan_task_vlm.py -> program.json) in AerialTransport with learned skills.

    # two arm-policy servers, one per robot (lerobot environment):
    python scripts/serve_lerobot_policy.py <ACT checkpoint> --port 6000 &
    python scripts/serve_lerobot_policy.py <ACT checkpoint> --port 6001 &
    # the executor (Isaac Sim environment):
    python scripts/run_transport_program.py --scene outputs/iris650_soarm100_AerialTransport_planning/ep0/scene.json \
        --program outputs/iris650_soarm100_AerialTransport_planning/ep0/qwen/program.json --arm-policy localhost:6000 localhost:6001 \
        --video outputs/iris650_soarm100_AerialTransport_planning/ep0/qwen/run.mp4

Nothing is scripted or simplified in the physics: both drones fly, the arms' dynamics act on the airframes
(no ``freeze_arm_dynamics``), every contact is on, and the box is held by the grippers' contact with its handle
bars alone (no kinematic attachment, no weight feedforward).

* drone: the world-model-trained CTBR policy (scripts/train_aerial_waypoint_wm.py), one copy per robot. It is
  given the reference point of the robot's current subtask: moving along the segment for goto / carry, fixed for
  hover and while waiting for another robot.
* arm: `grasp` runs the ACT policy at 25 Hz from the robot's own cameras; `stow` holds the default posture,
  `hold` keeps the last joint targets (jaw closed on the bar).
* a subtask ends on its `done_when`; one starts when everything it depends on has ended. The run stops before
  the first `release` subtask (--stop-at), i.e. after the box has been carried to the target.

Writes <video>, and next to it run.json (per-subtask outcome, tracking errors, where the box ended up) and
run.npz (time series).
"""
import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--scene", required=True)
p.add_argument("--program", required=True)
p.add_argument("--drone-policy", default=str(REPO / "outputs/iris650_soarm100_AerialWaypointTrack/wm_noarm/actor.pt"),
               help="actor trained with task.arm_obs=none (it is not shown the arm)")
p.add_argument("--arm-policy", nargs="+", default=None, metavar="HOST:PORT",
               help="one scripts/serve_lerobot_policy.py server per robot (omit: stop when the first grasp is due)")
p.add_argument("--stop-at", default="release", help="arm skill at which the run stops ('' = run everything)")
p.add_argument("--grasp-seconds", type=float, default=8.0, help="how long the arm policy runs (its demos take 7 s)")
p.add_argument("--timeout", type=float, default=6.0, help="seconds a subtask may overrun its nominal time before the run aborts")
p.add_argument("--video", default=None)
p.add_argument("--video-every", type=int, default=4, help="control steps per video frame")
p.add_argument("--image-size", type=int, nargs=2, default=[320, 240])
args = p.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=[
        "task=AerialTransport", "headless=true", "++task.env.num_envs=1", "++task.env.max_episode_length=1000000",
        "++task.sim.dt=0.004", "++task.sim.substeps=4", "++task.drone_action=ctbr"])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)

try:
    import numpy as np
    import torch
    from torch import nn
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation as R

    from omni_drones.utils.torch import euler_to_quaternion, quat_rotate, quat_rotate_inverse, quaternion_to_rotation_matrix
    from aetos_multiam.environments import make as make_env
    from aetos_multiam.environments.single.aerial_waypoint_track import (
        ARM_LOOKAHEAD, EXO_DIM, LOOKAHEAD, S_POS, S_RATE, S_ROT, S_VEL, STATE_DIM, X_REF_POS, X_REF_VEL, policy_obs)
    from aetos_multiam.planning.compile import segment_ref

    scene = json.loads(Path(args.scene).read_text())
    program = json.loads(Path(args.program).read_text())
    subtasks = {e["id"]: e for e in program["subtasks"]}
    robots = program["robots"]
    assert robots == [f"am_{i}" for i in range(len(robots))], robots

    env = make_env("AerialTransport", cfg, headless=True)
    td = env.reset()
    dev, origin = env.device, env.envs_positions[0]
    dt = float(env.sim.get_physics_dt()) * env.substeps
    stage = env.sim.stage if hasattr(env.sim, "stage") else __import__("omni.usd").usd.get_context().get_stage()
    box_view = env.carry_box._view
    z = np.load(REPO / "aetos_multiam/data/aerial_transport/so101_feasible_bases.npz")
    Q_STOW = torch.tensor([*z["q_default"], 1.5], dtype=torch.float32, device=dev)  # default posture, jaw open
    GRASP_LOCAL = torch.tensor([[0.0108, -0.0004, -0.0682]], device=dev)
    DEPTH_MAX = {"wrist": 0.5, "base": 1.0}
    G = 9.81

    # -- robots ------------------------------------------------------------------------------------------
    def look_at(eye, target, up):
        zc = np.asarray(eye, float) - target
        zc /= np.linalg.norm(zc)
        x = np.cross(up, zc)
        x /= np.linalg.norm(x)
        return np.roll(R.from_matrix(np.stack([x, np.cross(zc, x), zc], 1)).as_quat(), 1)

    env_prim = stage.GetPrimAtPath("/World/envs/env_0")
    grippers = sorted(x.GetPath().pathString for x in Usd.PrimRange(env_prim)
                      if x.GetName() == "gripper" and x.HasAPI(UsdPhysics.RigidBodyAPI))
    bots = []
    for i, am in enumerate(env.aerial_manipulators):
        view = am.drone._view
        idx = am.teleop_joint_indices.to(dev)
        # As the drone policy and the arm policy were trained: shoulder_pan's and the jaw's authored drives are too soft.
        view.set_gains(kps=torch.full((1, 1), 24.0, device=dev), kds=torch.full((1, 1), 3.4, device=dev), joint_indices=idx[0:1])
        view.set_gains(kps=torch.full((1, 1), 10.0, device=dev), kds=torch.full((1, 1), 0.5, device=dev), joint_indices=idx[5:6])
        names = list(view.body_names)
        rc = env.rate_controllers[i]
        hover = float(am.drone.params["mass"]) * G / rc.max_total_thrust
        b = dict(name=robots[i], am=am, drone=am.drone, view=view, idx=idx, g=names.index("gripper"), L=len(names), rc=rc,
                 hover=hover, last_action=torch.tensor([[0.0, 0.0, 0.0, 2 * hover - 1]], device=dev),
                 arm_target=Q_STOW.clone(), cur=None, t0=0.0, ref=None, cams={}, conn=None, act_k=-1, odom0=None)
        if args.arm_policy:
            from isaacsim.sensors.camera import Camera

            gp = [g for g in grippers if f"/{'am_' + str(i)}" in g or True][i]
            root = gp.rsplit("/", 1)[0]
            while not stage.GetPrimAtPath(f"{root}/base_link").IsValid():
                root = root.rsplit("/", 1)[0]
            defs = {  # as scripts/collect_arm_grasp_demo.py (--wrist-view forward), the views the arm policy was trained on
                "wrist": (gp, (-0.12, 0.0, 0.03), look_at((-0.12, 0.0, 0.03), np.array((0.011, 0.0, -0.10)), (-1.0, 0.0, 0.0))),
                "base": (f"{root}/base_link", (0.17458, 0.00309, -0.05512),
                         np.roll(R.from_euler("xyz", (1.857, 9.042, 44.59), degrees=True).as_quat(), 1)),
            }
            for n, (parent, eye, quat) in defs.items():
                cam = UsdGeom.Camera.Define(stage, f"{parent}/{n}_cam")
                cam.AddTranslateOp().Set(Gf.Vec3d(*eye))
                cam.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat]))
                cam.CreateFocalLengthAttr(12.0)
                cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
                b["cams"][n] = Camera(prim_path=f"{parent}/{n}_cam", resolution=tuple(args.image_size))
            from multiprocessing.connection import Client

            host, port = args.arm_policy[i].split(":")
            b["conn"] = Client((host, int(port)))
        bots.append(b)
    for b in bots:
        for c in b["cams"].values():
            c.initialize()
            c.add_distance_to_image_plane_to_frame()

    actor = nn.Sequential(nn.Linear(78, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 8)).to(dev)
    actor.load_state_dict(torch.load(args.drone_policy, map_location=dev))

    def drone_state(b):
        return b["drone"].get_state()[0, 0, :13]  # position relative to the env origin

    def ee_pos(b) -> np.ndarray:
        lt = b["view"]._physics_view.get_link_transforms().reshape(-1, b["L"], 7)[0, b["g"]]
        q = torch.cat([lt[6:7], lt[3:6]]).reshape(1, 4)
        return (lt[:3] + quat_rotate(q, GRASP_LOCAL)[0] - origin).cpu().numpy()

    def joints(b):
        return b["view"].get_joint_positions().reshape(-1, b["view"].num_dof)[0, b["idx"]]

    def box_pose():
        pos, quat = box_view.get_world_poses()
        return (pos.reshape(3) - origin).cpu().numpy(), R.from_quat(np.roll(quat.reshape(4).cpu().numpy(), -1))

    def handle_error(b, handle: str):
        """The gripper centre in the handle's frame: (distance off the bar's axis, slide along the bar), m."""
        bp, br = box_pose()
        rel = br.inv().apply(ee_pos(b) - bp) - np.array(program["handles"][handle]["local"])
        return math.hypot(rel[0], rel[2]), abs(rel[1])

    def held(b, handle: str, jaw: bool = True) -> bool:
        """scripts/collect_arm_grasp_demo.py's criterion: the bar is between the fingers, clear of its posts, and
        (jaw=True, the verdict on a grasp) the jaw stalled on it instead of closing. While carrying only the
        geometry counts: the bar settles deeper into the gripper under load and the jaw closes behind it."""
        off, slide = handle_error(b, handle)
        return (not jaw or float(joints(b)[5]) > 0.05) and off < 0.03 and slide < 0.10

    # -- initial state: as the planner was told ------------------------------------------------------------
    for b, r in zip(bots, scene["robots"]):
        d, n = b["drone"], b["view"].num_dof
        d.set_world_poses((torch.tensor(r["position"], device=dev, dtype=torch.float32) + origin).reshape(1, 3),
                          euler_to_quaternion(torch.tensor([[0.0, 0.0, math.radians(r["yaw_deg"])]], device=dev)))
        d.set_velocities(torch.zeros(1, 6, device=dev))
        b["view"].set_joint_velocities(torch.zeros(1, 1, n, device=dev), joint_indices=torch.arange(n, device=dev))
        b["view"].set_joint_positions(Q_STOW.reshape(1, 1, 6), joint_indices=b["idx"])
        b["view"].set_joint_position_targets(Q_STOW.reshape(1, 1, 6), joint_indices=b["idx"])
        d.throttle[:] = math.sqrt(b["hover"])
        b["rc"].reset()
        b["ref"] = (np.array(r["position"], float), math.radians(r["yaw_deg"]))
    o = scene["objects"][0]
    box_view.set_world_poses((torch.tensor(o["position"], device=dev, dtype=torch.float32) + origin).reshape(1, 3),
                             euler_to_quaternion(torch.tensor([[0.0, 0.0, math.radians(o["yaw_deg"])]], device=dev)))
    box_view.set_velocities(torch.zeros(1, 6, device=dev))
    target = next(x for x in scene["objects"] if x["name"] == "drop_zone")

    vp = frames_dir = None
    if args.video:
        import shutil
        import subprocess

        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        vp = get_active_viewport()
        cs = ViewportCameraState("/OmniverseKit_Persp", vp)
        cs.set_position_world(Gf.Vec3d(float(origin[0]) + 3.2, float(origin[1]) - 3.6, 3.4), True)
        cs.set_target_world(Gf.Vec3d(float(origin[0]), float(origin[1]) + 1.2, 0.7), True)
        frames_dir = Path(args.video).with_suffix("")
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True)
    for _ in range(10):
        env.sim.render()

    # -- skills --------------------------------------------------------------------------------------------
    HORIZON = max(max(LOOKAHEAD), max(ARM_LOOKAHEAD)) + 1

    def reference(b, t, k=0):
        """(position, yaw) of the robot's reference k control steps from now."""
        e = b["cur"]
        if e is None:
            return b["ref"]
        return segment_ref(program, e, t - b["t0"] + k * dt)

    def drone_action(b, t):
        st = drone_state(b)
        s = torch.zeros(1, STATE_DIM, device=dev)
        s[0, S_POS], s[0, S_ROT] = st[:3], quaternion_to_rotation_matrix(st[3:7]).reshape(9)
        s[0, S_VEL], s[0, S_RATE] = st[7:10], quat_rotate_inverse(st[3:7].reshape(1, 4), st[10:13].reshape(1, 3))[0]
        exo = torch.zeros(1, HORIZON, EXO_DIM, device=dev)
        pts = {k: reference(b, t, k) for k in LOOKAHEAD}
        for k, (pos, _) in pts.items():
            exo[0, k, X_REF_POS] = torch.as_tensor(pos, dtype=torch.float32, device=dev)
        vel = (reference(b, t, 1)[0] - pts[0][0]) / dt
        exo[0, 0, X_REF_VEL] = torch.as_tensor(vel, dtype=torch.float32, device=dev)
        yaw = torch.tensor([pts[0][1]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = torch.tanh(actor(policy_obs(s, b["last_action"], exo, yaw, "none"))[:, :4])
        b["last_action"] = a
        return a, float(np.linalg.norm(pts[0][0] - st[:3].cpu().numpy())), float(st[7:10].norm())

    def images(b):
        out = {}
        for n, c in b["cams"].items():
            out[n] = np.ascontiguousarray(np.asarray(c.get_rgba())[:, :, :3])
        for n, c in b["cams"].items():
            dpt = np.nan_to_num(np.asarray(c.get_depth(), dtype=np.float32), nan=np.inf, posinf=np.inf)
            out[f"{n}_depth"] = np.repeat((np.clip(dpt / DEPTH_MAX[n], 0.0, 1.0) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        return out

    def arm_policy_step(b):
        """One 25 Hz step of the arm policy: joints + the drone's own odometry + its cameras -> joint targets."""
        st = drone_state(b).cpu().numpy()
        rot = R.from_quat(np.roll(st[3:7], -1))
        pos, yaw = st[:3].astype(float), float(rot.as_euler("zyx")[0])
        if b["odom0"] is None:
            b["odom0"] = (pos, yaw)
            b["conn"].send("reset")
        p0, y0 = b["odom0"]
        dstate = [*R.from_euler("z", -y0).apply(pos - p0), (yaw - y0 + math.pi) % (2 * math.pi) - math.pi,
                  *rot.inv().apply(st[7:10]), rot.inv().apply(st[10:13])[2]]
        b["conn"].send({"state": joints(b).tolist() + [float(v) for v in dstate],
                        "shape": (args.image_size[1], args.image_size[0]),
                        **{n: im.tobytes() for n, im in images(b).items()}})
        b["arm_target"] = torch.tensor(b["conn"].recv()[:6], dtype=torch.float32, device=dev)

    # -- run -----------------------------------------------------------------------------------------------
    # which handle each robot is to hold, for the log: the handle its first grasp names
    plan_alloc = [dict(robot=e["robot"], part=e["arm"]["object"]) for e in program["subtasks"] if e["arm"]["skill"] == "grasp"]
    status = {i: "pending" for i in subtasks}
    events, log = [], {k: [] for k in ("t", "box_pos", "box_tilt_deg", *(f"{r}_{q}" for r in robots for q in ("pos", "ref", "err", "jaw", "bar_off", "bar_slide")))}
    t, step, frame, outcome = 0.0, 0, 0, "completed"
    box_z0 = box_pose()[0][2]

    def ready(e):
        return all(status[d] == "done" for d in e["deps"])

    def finish(e, how):
        status[e["id"]] = "done"
        b = bots[robots.index(e["robot"])]
        b["ref"], b["cur"] = segment_ref(program, e, 1e9), None
        events.append(dict(id=e["id"], robot=e["robot"], start=round(e["_t0"], 2), end=round(t, 2), nominal_s=round(e["drone"]["duration_s"], 2),
                           result=how, pos_err_cm=round(e.get("_err", 0.0) * 100, 1)))
        print(f"[run] {t:6.2f}s {e['robot']} {e['id']:14s} {how} (took {t - e['_t0']:.1f}s, nominal {e['drone']['duration_s']:.1f}s)", flush=True)

    while outcome == "completed":
        # start what can start; members of a sync group start together
        stop = False
        for e in program["subtasks"]:
            b = bots[robots.index(e["robot"])]
            if status[e["id"]] != "pending" or b["cur"] is not None or not ready(e):
                continue
            mine = [x for x in program["subtasks"] if x["robot"] == e["robot"] and status[x["id"]] == "pending"]
            if mine[0] is not e:
                continue
            if args.stop_at and e["arm"]["skill"] == args.stop_at:
                stop = True
                continue
            if e["arm"]["skill"] == "grasp" and b["conn"] is None:
                outcome = "stopped: no arm policy given"
                break
            group = [x for x in program["subtasks"] if e["sync_group"] and x["sync_group"] == e["sync_group"]] or [e]
            if any(bots[robots.index(x["robot"])]["cur"] is not None for x in group):
                continue
            for x in group:
                bx = bots[robots.index(x["robot"])]
                bx["cur"], bx["t0"], x["_t0"], status[x["id"]] = x, t, t, "running"
                bx["odom0"], bx["act_k"] = None, -1
                if x["arm"]["skill"] == "stow":
                    bx["arm_target"] = Q_STOW.clone()
                print(f"[run] {t:6.2f}s {x['robot']} {x['id']:14s} start: drone {x['drone']['skill']} ({x['drone']['mode']}), arm {x['arm']['skill']}", flush=True)
        if outcome != "completed" or (not any(b["cur"] for b in bots) and (stop or all(s == "done" for s in status.values()))):
            break
        if not any(b["cur"] for b in bots):
            outcome = "deadlock: nothing can start"
            break

        # arm policies: one step every 40 ms of simulated time, from freshly rendered cameras
        acting = [b for b in bots if b["cur"] and b["cur"]["arm"]["skill"] == "grasp" and int((t - b["t0"]) / 0.04) > b["act_k"]
                  and t - b["t0"] < args.grasp_seconds]
        if acting:
            env.sim.render()
            for b in acting:
                b["act_k"] = int((t - b["t0"]) / 0.04)
                arm_policy_step(b)

        errs = {}
        for i, b in enumerate(bots):
            a, err, speed = drone_action(b, t)
            errs[b["name"]] = (err, speed)
            act = torch.zeros(env.action_spec[(f"agent_{i}", "action")].shape, device=dev)
            act[0, 0, :4] = a[0]
            td.set((f"agent_{i}", "action"), act)
            b["view"].set_joint_position_targets(b["arm_target"].reshape(1, 1, 6), joint_indices=b["idx"])
        td = env.step(td)["next"]
        t, step = t + dt, step + 1

        bp, br = box_pose()
        log["t"].append(t)
        log["box_pos"].append(bp)
        log["box_tilt_deg"].append(math.degrees(math.acos(np.clip(br.as_matrix()[2, 2], -1, 1))))
        for b in bots:
            st = drone_state(b)[:3].cpu().numpy()
            log[f"{b['name']}_pos"].append(st)
            log[f"{b['name']}_ref"].append(reference(b, t)[0])
            log[f"{b['name']}_err"].append(errs[b["name"]][0])
            hname = next((a["part"] for a in plan_alloc if a["robot"] == b["name"]), None)
            off, slide = handle_error(b, hname) if hname else (0.0, 0.0)
            log[f"{b['name']}_jaw"].append(float(joints(b)[5]))
            log[f"{b['name']}_bar_off"].append(off)
            log[f"{b['name']}_bar_slide"].append(slide)
            if not np.isfinite(st).all() or st[2] < 0.25:
                outcome = f"{b['name']} crashed"
        if frames_dir is not None and step % args.video_every == 0:
            env.sim.render()
            capture_viewport_to_file(vp, str(frames_dir / f"{frame:05d}.png"))
            frame += 1

        # what has ended; a sync group ends together
        for e in [x for x in program["subtasks"] if status[x["id"]] == "running"]:
            b = bots[robots.index(e["robot"])]
            tau, nominal = t - b["t0"], e["drone"]["duration_s"]
            e["_err"] = errs[b["name"]][0]
            if e["done_when"] == "grasped":
                if tau >= args.grasp_seconds:
                    if held(b, e["arm"]["object"]):
                        e["_ok"] = True
                    else:
                        off, slide = handle_error(b, e["arm"]["object"])
                        finish(e, f"GRASP FAILED (bar {off * 1000:.0f} mm off the gripper centre, slid {slide * 1000:.0f} mm, jaw {float(joints(b)[5]):.2f} rad)")
                        outcome = f"{e['id']}: grasp failed"
            elif tau >= nominal and errs[b["name"]][0] < 0.05 and errs[b["name"]][1] < 0.15:
                e["_ok"] = True
            elif tau > nominal + args.timeout:
                finish(e, f"TIMEOUT ({errs[b['name']][0] * 100:.1f} cm from the reference)")
                outcome = f"{e['id']}: timed out"
        for e in [x for x in program["subtasks"] if status[x["id"]] == "running" and x.get("_ok")]:
            group = [x for x in program["subtasks"] if e["sync_group"] and x["sync_group"] == e["sync_group"]] or [e]
            if all(x.get("_ok") for x in group):
                finish(e, "done")
        # a robot that holds a handle must keep holding it
        for b in bots:
            e = b["cur"]
            if e is not None and e["arm"]["skill"] == "hold" and not held(b, e["arm"]["object"], jaw=False):
                off, slide = handle_error(b, e["arm"]["object"])
                finish(e, f"LOST THE HANDLE (bar {off * 1000:.0f} mm off the gripper centre, slid {slide * 1000:.0f} mm)")
                outcome = f"{e['id']}: lost {e['arm']['object']}"

    bp, br = box_pose()
    result = dict(
        outcome=outcome, sim_time_s=round(t, 2), subtasks=events,
        not_run=[i for i, s in status.items() if s == "pending"],
        tracking_error_cm={r: dict(mean=round(float(np.mean(log[f"{r}_err"])) * 100, 2), max=round(float(np.max(log[f"{r}_err"])) * 100, 2)) for r in robots},
        box=dict(final_pos=np.round(bp, 3).tolist(), final_yaw_deg=round(float(br.as_euler("zyx", degrees=True)[0]), 1),
                 xy_error_to_target_cm=round(float(np.linalg.norm(bp[:2] - np.array(target["position"][:2]))) * 100, 1),
                 height_above_start_cm=round(float(bp[2] - box_z0) * 100, 1), max_tilt_deg=round(float(np.max(log["box_tilt_deg"])), 1),
                 held_by={b["name"]: bool(any(held(b, h, jaw=False) for h in program["handles"])) for b in bots}))
    out = Path(args.video).parent if args.video else Path(args.program).parent
    (out / "run.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out / "run.npz", **{k: np.asarray(v) for k, v in log.items()})
    print("[run] " + json.dumps({k: v for k, v in result.items() if k != "subtasks"}), flush=True)
    if frames_dir is not None and frame:
        import imageio_ffmpeg

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(round(1 / (dt * args.video_every))),
                        "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.video], check=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[run] video: {args.video} ({frame} frames)", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
