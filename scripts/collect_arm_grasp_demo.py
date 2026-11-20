"""Collect SO101 grasp demonstrations from random drone-base poses (arm only, real contact).

The drone is bolted to a moving mount instead of flying: each episode the mount is put at one
feasible base pose (scripts/arm_planning/feasible_base_set.py) and then drifts a little, the way
a hovering drone does. The arm starts in its default posture and a scripted expert drives the
gripper centre -- targets are poses in the world, re-solved every step by the Cartesian
controller -- to a pregrasp, onto the handle bar, closes the jaw, and lifts the handle. All
contacts are on; an episode counts (and is saved) only if the bar is still in the jaw after the lift.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python scripts/collect_arm_grasp_demo.py --episodes 20 --out outputs/arm_grasp_demos

Closed-loop test of a trained policy in the same scene, on base poses the training set never used
(start the server first, in the lerobot environment: scripts/serve_lerobot_policy.py):

    python scripts/collect_arm_grasp_demo.py --policy localhost:6000 --episodes 50 --seed 100 \\
        --exclude-bases-from outputs/arm_grasp_demos

One npz per episode, 25 Hz: images (wrist, base cameras), joint state, the expert's clean joint
targets (``action``) and the noisy ones actually sent (``action_executed``), phase labels, base /
box / gripper-centre poses, success.
"""
import argparse
import math
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--bases", default=str(REPO / "aetos_multiam/data/aerial_transport/so101_feasible_bases.npz"))
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--out", default=None, help="directory for the episode files (omit: only report success)")
p.add_argument("--no-cameras", action="store_true")
p.add_argument("--image-size", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
p.add_argument("--jitter", type=float, default=0.01, help="size of the base's slow hover drift, m per axis (yaw: 1 deg per cm)")
p.add_argument("--action-noise", type=float, default=0.003,
               help="std of the noise added to the arm targets sent, rad; redrawn at the recording rate "
                    "(0.01 redrawn every physics step shook the arm off the bar: 3/14 grasps)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--policy", default=None, metavar="HOST:PORT",
               help="evaluate a trained policy (scripts/serve_lerobot_policy.py) instead of the scripted expert")
p.add_argument("--exclude-bases-from", default=None, metavar="DIR",
               help="skip the base poses used by the episodes under DIR (the training set), to test on unseen ones")
p.add_argument("--base-yaw-max", type=float, default=None, metavar="DEG", help="only base poses whose heading is within +-DEG of the handle frame's")
p.add_argument("--base-height", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="only base poses this far above the handle, m")
p.add_argument("--base-xy-range", type=float, default=None, metavar="M", help="only base poses within +-M of the nominal base in x and y (handle frame)")
p.add_argument("--base-state", action="store_true",
               help="with --policy: the policy was trained on export_arm_grasp_lerobot.py --base-state")
p.add_argument("--drone-state", action="store_true",
               help="with --policy: append the drone's own state (DRONE_STATE_NAMES) to the joint state sent, "
                    "for a policy trained on export_arm_grasp_lerobot.py --drone-state")
p.add_argument("--replay", default=None, metavar="DATASET",
               help="replay the recorded actions of a LeRobot dataset (scripts/export_arm_grasp_lerobot.py) open loop "
                    "and report how many episodes still succeed; needs --replay-init")
p.add_argument("--replay-init", default=None, metavar="NPZ",
               help="per dataset episode: base_index, box_xy_yaw_deg, drift_amp, drift_freq, drift_phase "
                    "(what the collector drew for that episode -- the dataset itself does not store it)")
p.add_argument("--replay-interp", action="store_true",
               help="ramp between consecutive recorded targets over the physics steps instead of holding each one")
p.add_argument("--depth", action="store_true",
               help="also record / send each camera's depth image as <camera>_depth: distance to the image plane, "
                    "0..DEPTH_MAX m mapped to 0..255 (farther or no return = 255), as a three-channel image")
p.add_argument("--wrist-view", choices=["forward", "side"], default="forward", help="where the wrist camera looks (side: the older datasets)")
p.add_argument("--handle", type=int, choices=[0, 1], default=0,
               help="which handle of the box to grasp: 0 (-x, where the demos were recorded) or 1 (+x, the second robot's, "
                    "the same geometry turned half a turn about the vertical)")
p.add_argument("--video", default=None, metavar="DIR", help="save every episode's camera views (side by side) as an mp4, failures too")
p.add_argument("--gui", action="store_true")
args = p.parse_args()
if args.policy or args.replay:
    args.action_noise = 0.0  # the policy's / the recorded commands go out as they are

from aetos_multiam import init_simulation_app  # noqa: E402

GRASP_LOCAL = (0.0108, -0.0004, -0.0682)  # gripper centre on the 'gripper' link; = aerial_transport_track.GRASP_LOCAL
HANDLE_IN_BOX = (-0.42, 0.0, 0.34)  # bar centre of handle 0 (CarryBox.handle_local_offsets)
TABLE_Z = 0.75
JAW_OPEN, JAW_KP, JAW_KD = 1.5, 10.0, 0.5  # the authored jaw drive (kp 0.35) cannot hold anything
RETREAT, LIFT = 0.08, 0.04  # m: pregrasp distance back along the fingers; how far the handle is lifted
PHASES = (("reach", 2.0), ("approach", 1.5), ("close", 1.0), ("lift", 1.5), ("hold", 1.0))
DT, RECORD_EVERY = 0.01, 4
DEPTH_MAX = {"wrist": 0.5, "base": 1.0}  # m, the range a depth image covers (--depth): the wrist camera only needs the bar
# What the drone knows about itself without seeing the scene (odometry + IMU), nothing about the box:
# displacement since the episode started, in the heading frame it started with; heading change;
# linear velocity and yaw rate in its own body frame.
DRONE_STATE_NAMES = ("drone_dx", "drone_dy", "drone_dz", "drone_dyaw", "drone_vx", "drone_vy", "drone_vz", "drone_wz")

simulation_app = init_simulation_app({"headless": not args.gui})
try:
    import torch
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import RigidPrim as RigidPrimView
    from isaacsim.core.api.simulation_context import SimulationContext
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    from scipy.spatial.transform import Rotation as R

    import omni_drones.utils.kit as kit_utils

    from base_state import base_state  # scripts/base_state.py

    from aetos_multiam.controllers.arm import CartesianController, EndEffector, pose_error
    from aetos_multiam.objects.carry_box import CarryBox
    from aetos_multiam.robots.presets import get_preset
    from aetos_multiam.robots.robot_base import AerialManipulatorBase

    def T(pos, quat_wxyz=(1, 0, 0, 0)):
        m = np.eye(4)
        m[:3, :3] = R.from_quat(np.roll(np.asarray(quat_wxyz, float), -1)).as_matrix()
        m[:3, 3] = pos
        return m

    def pq(m):  # 4x4 -> (pos, quat wxyz) float32 tensors, batch of one
        return (torch.tensor(m[:3, 3], dtype=torch.float32).reshape(1, 3),
                torch.tensor(np.roll(R.from_matrix(m[:3, :3]).as_quat(), 1), dtype=torch.float32).reshape(1, 4))

    # Handle frame in the box frame; the second handle's is the first's turned half a turn about the box's vertical.
    HANDLE_T = T(HANDLE_IN_BOX) if args.handle == 0 else T((0, 0, 0), (0, 0, 0, 1)) @ T(HANDLE_IN_BOX)

    def look_at(eye, target, up=(0.0, 0.0, 1.0)):  # USD camera: looks along -z, +y up
        z = np.asarray(eye, float) - target
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        return np.roll(R.from_matrix(np.stack([x, np.cross(z, x), z], 1)).as_quat(), 1)

    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT, backend="torch",
        physics_prim_path="/World/PhysicsScene", device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, -9.81)},
    )
    stage = sim.stage
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(600.0)
    FixedCuboid("/World/table", translation=(0.0, 0.0, TABLE_Z - 0.02), scale=np.array([1.4, 0.9, 0.04]),
                color=np.array([0.55, 0.57, 0.6]))
    box = CarryBox(name="carry_box", translation=(0.0, 0.0, TABLE_Z + 0.002))
    box.spawn("/World")

    am = AerialManipulatorBase(get_preset("iris650_soarm100"), name="arm")
    BASE0 = (0.0, 0.0, 3.0)
    root = am.spawn(translations=[BASE0])[0].GetPath().pathString
    # The drone hangs from a kinematic mount by a fixed joint kept outside the articulation (which so
    # stays floating-base: the Jacobian keeps the layout EndEffector reads). Moving the mount moves the base.
    mount = UsdGeom.Xform.Define(stage, "/World/mount")
    mount.AddTranslateOp().Set(Gf.Vec3d(*BASE0))
    UsdPhysics.RigidBodyAPI.Apply(mount.GetPrim()).CreateKinematicEnabledAttr(True)
    pin = UsdPhysics.FixedJoint.Define(stage, "/World/pin")
    pin.CreateBody0Rel().SetTargets(["/World/mount"])
    pin.CreateBody1Rel().SetTargets([f"{root}/base_link"])
    pin.CreateExcludeFromArticulationAttr(True)

    cams = {}
    if not args.no_cameras:
        from isaacsim.sensors.camera import Camera

        gripper_prim = [x for x in Usd.PrimRange(stage.GetPrimAtPath(root))
                        if x.GetName() == "gripper" and x.HasAPI(UsdPhysics.RigidBodyAPI)][0].GetPath().pathString
        # name: (parent prim, position, orientation wxyz), in the parent link's frame.
        cam_defs = {
            # forward: behind the fixed finger, looking along the fingers at what is ahead of them (gripper at the
            # bottom of the image). side: beside the fixed finger looking across at the gripper centre, as the
            # datasets recorded before this view existed.
            "wrist": (gripper_prim, (-0.12, 0.0, 0.03), look_at((-0.12, 0.0, 0.03), np.array((0.011, 0.0, -0.10)), (-1.0, 0.0, 0.0)))
            if args.wrist_view == "forward" else
            (gripper_prim, (-0.10, 0.0, -0.01), look_at((-0.10, 0.0, -0.01), np.asarray(GRASP_LOCAL), (0.0, 0.0, 1.0))),
            # Under the airframe ahead of the arm, looking down and slightly forward, turned ~45 deg
            # about its axis (set by hand in the viewport; Euler xyz, deg, as the Property panel shows it).
            "base": (f"{root}/base_link", (0.17458, 0.00309, -0.05512),
                     np.roll(R.from_euler("xyz", (1.857, 9.042, 44.59), degrees=True).as_quat(), 1)),
        }
        for name, (parent, eye, quat) in cam_defs.items():
            cam = UsdGeom.Camera.Define(stage, f"{parent}/{name}_cam")
            cam.AddTranslateOp().Set(Gf.Vec3d(*eye))
            cam.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat]))
            cam.CreateFocalLengthAttr(12.0)  # with the default 20.955 mm aperture: 82 deg across
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
            cams[name] = Camera(prim_path=f"{parent}/{name}_cam", resolution=tuple(args.image_size))

    sim.reset()
    am.initialize()
    for c in cams.values():
        c.initialize()
        if args.depth:
            c.add_distance_to_image_plane_to_frame()

    def images():  # what the cameras see now: name -> HxWx3 uint8 (colour, and with --depth the depth image)
        out = {n: c.get_rgba()[:, :, :3].copy() for n, c in cams.items()}
        if args.depth:
            for n, c in cams.items():
                d = np.nan_to_num(np.asarray(c.get_depth(), dtype=np.float32), nan=np.inf, posinf=np.inf)
                d8 = (np.clip(d / DEPTH_MAX[n], 0.0, 1.0) * 255).astype(np.uint8)
                out[f"{n}_depth"] = np.repeat(d8[:, :, None], 3, axis=2)
        return out
    image_names = [*cams, *(f"{n}_depth" for n in cams if args.depth)]
    view = am.drone._view
    mount_view = RigidPrimView("/World/mount", reset_xform_properties=False)
    box_view = RigidPrimView("/World/carry_box", reset_xform_properties=False)
    mount_view.initialize()
    box_view.initialize()

    arm_idx = torch.as_tensor(am.teleop_joint_indices)[:5]
    jaw_idx = torch.as_tensor(am.teleop_joint_indices)[5:6]
    all_idx = torch.cat([arm_idx, jaw_idx])
    dof_names = list(view._dof_names)
    view.set_gains(kps=torch.full((1, 1), 24.0), kds=torch.full((1, 1), 3.4),  # as task.shoulder_pan_gains
                   joint_indices=torch.tensor([dof_names.index("shoulder_pan")]))
    view.set_gains(kps=torch.full((1, 1), JAW_KP), kds=torch.full((1, 1), JAW_KD), joint_indices=jaw_idx)
    ee = EndEffector(view, "gripper", arm_idx, GRASP_LOCAL)
    lim = view.get_dof_limits().reshape(-1, view.num_dof, 2)[0][arm_idx]
    ctl = CartesianController(kp=8.0, kr=8.0, damping=0.02, position_first=True)

    z = np.load(args.bases)
    bases, grasps = z["drone_in_handle"][z["feasible"]], z["grasp_in_handle_each"][z["feasible"]]
    q_default = torch.tensor(z["q_default"], dtype=torch.float32)
    rng = np.random.default_rng(args.seed)
    allowed = np.ones(len(bases), bool)  # indices below are into the full feasible set, as base_index in the episode files
    base_ids = np.arange(len(bases))  # row of `bases` -> index in the full feasible set, kept through the filtering below
    if args.base_yaw_max is not None:
        allowed &= np.abs(R.from_matrix(bases[:, :3, :3]).as_euler("zyx", degrees=True)[:, 0]) <= args.base_yaw_max
    if args.base_height is not None:
        allowed &= (bases[:, 2, 3] >= args.base_height[0]) & (bases[:, 2, 3] <= args.base_height[1])
    if args.base_xy_range is not None:
        allowed &= (np.abs(bases[:, :2, 3] - z["drone_in_handle_nominal"][:2, 3]) <= args.base_xy_range).all(1)
    if not allowed.all():
        print(f"[collect] base range: {int(allowed.sum())} of {len(bases)} feasible base poses allowed", flush=True)
    if args.exclude_bases_from:
        seen = {int(np.load(f)["base_index"]) for f in Path(args.exclude_bases_from).rglob("ep*.npz")}
        print(f"[collect] {len(seen)} base poses used under {args.exclude_bases_from} left out", flush=True)
        allowed[list(seen)] = False
    assert allowed.any(), "no base pose left"
    if not allowed.all():
        keep = np.where(allowed)[0]
        print(f"[collect] {len(keep)} base poses to draw from", flush=True)
        bases, grasps, base_ids = bases[keep], grasps[keep], keep
    conn = None
    if args.policy:
        from multiprocessing.connection import Client
        host, port = args.policy.split(":")
        conn = Client((host, int(port)))

    replay = None
    if args.replay:
        import pandas as pd
        frames = pd.concat([pd.read_parquet(f) for f in sorted(Path(args.replay, "data").rglob("*.parquet"))])
        replay = [np.stack(g["action"].values) for _, g in frames.sort_values("index").groupby("episode_index")]
        init = np.load(args.replay_init)
        assert len(init["base_index"]) == len(replay), "--replay-init does not belong to this dataset"
        args.episodes = min(args.episodes, len(replay)) if "--episodes" in __import__("sys").argv else len(replay)

    def joints():
        return view.get_joint_positions().reshape(-1, view.num_dof)[0, all_idx]

    odom0 = []  # [position, heading] of the drone when the episode's first observation was taken

    def drone_state():
        bp, bq = view.get_world_poses()
        vel = view.get_velocities().reshape(-1, 6)[0].numpy()
        rot = R.from_quat(np.roll(bq.reshape(4).numpy(), -1))
        pos, yaw = bp.reshape(3).numpy().astype(float), float(rot.as_euler("zyx")[0])
        if not odom0:
            odom0.extend([pos, yaw])
        d = R.from_euler("z", -odom0[1]).apply(pos - odom0[0])
        dyaw = (yaw - odom0[1] + math.pi) % (2 * math.pi) - math.pi
        return np.array([*d, dyaw, *rot.inv().apply(vel[:3]), rot.inv().apply(vel[3:])[2]], dtype=np.float32)

    def handle_pose():
        bp, bq = box_view.get_world_poses()
        return T(bp[0].numpy(), bq[0].numpy()) @ HANDLE_T

    def ee_pose():
        pos, quat = ee.pose()
        return T(pos[0].numpy(), quat[0].numpy())

    def run_episode(ep):
        i = int(rng.integers(len(bases)))
        box_world = T((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), TABLE_Z + 0.002),
                      np.roll(R.from_euler("z", rng.uniform(-20, 20), degrees=True).as_quat(), 1))
        if replay:
            i = int(np.where(base_ids == int(init["base_index"][ep]))[0][0])
            x, y, yaw = init["box_xy_yaw_deg"][ep]
            box_world = T((x, y, TABLE_Z + 0.002), np.roll(R.from_euler("z", yaw, degrees=True).as_quat(), 1))
        base = box_world @ HANDLE_T @ bases[i]
        box_view.set_world_poses(*pq(box_world))
        box_view.set_velocities(torch.zeros(1, 6))
        mount_view.set_world_poses(*pq(base))
        view.set_world_poses(*pq(base))
        view.set_velocities(torch.zeros(1, 1, 6))
        q_cmd = q_default.reshape(1, 5).clone()
        jaw_cmd = JAW_OPEN
        q0 = torch.cat([q_cmd[0], torch.tensor([jaw_cmd])]).reshape(1, 1, 6)
        view.set_joint_positions(q0, joint_indices=all_idx)
        view.set_joint_velocities(torch.zeros(1, 1, view.num_dof))
        view.set_joint_position_targets(q0, joint_indices=all_idx)

        # Hover drift of x y z (m) and yaw (rad): two slow sinusoids each (0.1-0.6 Hz), random per episode.
        scale = np.array([1, 1, 1, math.radians(100)]) * args.jitter
        amp, freq, phase0 = rng.normal(size=(2, 4)) * scale, rng.uniform(0.1, 0.6, (2, 4)), rng.uniform(0, 2 * np.pi, (2, 4))
        if replay:
            amp, freq, phase0 = init["drift_amp"][ep], init["drift_freq"][ep], init["drift_phase"][ep]
        t, noise = 0.0, torch.zeros(5)
        odom0.clear()
        rec = {k: [] for k in ("q", "action", "action_executed", "phase", "drone_state", "base_pose", "box_pose", "ee_pose",
                               "ee_target", *(f"image_{n}" for n in image_names))}

        mount_now = [base]  # where the mount was last sent

        def step(target, phase, k):
            nonlocal q_cmd, t
            t += DT
            drift = (amp * np.sin(2 * np.pi * freq * t + phase0)).sum(0)
            m = base @ T(drift[:3], np.roll(R.from_euler("z", drift[3]).as_quat(), 1))
            mount_now[0] = m
            bp, bq = pq(m)
            mount_view._physics_view.set_kinematic_targets(torch.cat([bp, bq[:, 1:], bq[:, :1]], -1), torch.tensor([0]))
            if target is not None:
                pos, quat = ee.pose()
                e_p, e_r = pose_error(*pq(target), pos, quat)
                q_now = view.get_joint_positions().reshape(-1, view.num_dof)[:, arm_idx]
                q_cmd = ctl.step(ee.jacobian(), e_p, e_r, q_now, q_cmd, lim[:, 0], lim[:, 1], DT)
            clean = torch.cat([q_cmd[0], torch.tensor([jaw_cmd])])
            if k % RECORD_EVERY == 0:
                noise[:] = args.action_noise * torch.randn(5)
            sent = clean.clone()
            sent[:5] += noise
            view.set_joint_position_targets(sent.reshape(1, 1, 6), joint_indices=all_idx)
            if (args.out or args.video) and phase >= 0 and k % RECORD_EVERY == 0:
                for n, im in images().items():
                    rec[f"image_{n}"].append(im)
                bpos, bquat = box_view.get_world_poses()
                for key, v in (("q", joints().numpy()), ("drone_state", drone_state()), ("action", clean.numpy()), ("action_executed", sent.numpy()),
                               ("phase", phase), ("base_pose", m), ("box_pose", T(bpos[0].numpy(), bquat[0].numpy())),
                               ("ee_pose", ee_pose()), ("ee_target", target if target is not None else ee_pose())):
                    rec[key].append(np.asarray(v, dtype=np.float32 if key != "phase" else np.int64))
            sim.step(render=bool(cams) and bool((args.out and not replay) or conn or args.video) and (k + 1) % RECORD_EVERY == 0 or args.gui)

        for k in range(50):  # settle in the default posture
            step(None, -1, k)
        h0 = handle_pose()
        grasp = h0 @ grasps[i]  # the gripper-centre pose this base's plan ends in, on the live handle
        pre = grasp @ T((0.0, 0.0, RETREAT))
        start = ee_pose()
        k, left = 0, []
        if conn:  # the policy drives arm and jaw at the recording rate, for the expert's time plus 1 s
            conn.send("reset")
            m_prev = mount_now[0]
            for k in range(int((sum(s for _, s in PHASES) + 1.0) / DT)):
                if k % RECORD_EVERY == 0:
                    state = joints().tolist()
                    if args.base_state:
                        state += base_state(m_prev, mount_now[0], DT * RECORD_EVERY).tolist()
                        m_prev = mount_now[0]
                    if args.drone_state:
                        state += drone_state().tolist()
                    conn.send({"state": state, "shape": (args.image_size[1], args.image_size[0]),
                               **{n: im.tobytes() for n, im in images().items()}})
                    a = conn.recv()
                    q_cmd, jaw_cmd = torch.tensor(a[:5], dtype=torch.float32).reshape(1, 5), float(a[5])
                step(None, 0, k)
        if replay:  # the recorded targets, one per recorded frame
            acts = torch.tensor(replay[ep], dtype=torch.float32)
            for k in range(len(acts) * RECORD_EVERY):
                f, u = divmod(k, RECORD_EVERY)
                a = acts[f]
                if args.replay_interp and f + 1 < len(acts):
                    a = a + (acts[f + 1] - a) * u / RECORD_EVERY
                q_cmd, jaw_cmd = a[:5].reshape(1, 5).clone(), float(a[5])
                step(None, 0, k)
        for phase, (name, seconds) in enumerate(() if conn or replay else PHASES):
            n = int(seconds / DT)
            if name == "lift":
                lift_from = ee_pose()
                lift_from[:3, :3] = grasp[:3, :3]
            for j in range(n):
                u = (j + 1) / n
                u = u * u * (3 - 2 * u)
                target = grasp.copy()
                if name == "reach":
                    target = pre.copy()
                    target[:3, 3] = start[:3, 3] + u * (pre[:3, 3] - start[:3, 3])
                elif name == "approach":
                    target[:3, 3] = pre[:3, 3] + u * (grasp[:3, 3] - pre[:3, 3])
                elif name == "close":
                    jaw_cmd = JAW_OPEN * (1 - u)
                elif name in ("lift", "hold"):
                    target = lift_from.copy()
                    target[2, 3] += LIFT * (u if name == "lift" else 1.0)
                step(target, phase, k)
                k += 1
            miss = np.linalg.inv(target) @ ee_pose()
            left.append(f"{name} {np.linalg.norm(miss[:3, 3]) * 1000:.0f} mm/{np.degrees(np.linalg.norm(R.from_matrix(miss[:3, :3]).as_rotvec())):.0f} deg")
            if name == "close":  # squeeze with a fixed preload instead of driving the jaw through the bar
                jaw_cmd = max(float(joints()[5]) - 0.2, 0.0)

        rel = np.linalg.inv(handle_pose()) @ ee_pose()  # gripper centre in the handle frame (bar along y)
        off_axis, slide = math.hypot(rel[0, 3], rel[2, 3]), abs(rel[1, 3])
        lifted = handle_pose()[2, 3] - h0[2, 3]
        # Held: the jaw stalled on the bar instead of closing (0 rad), the bar is between the fingers and clear
        # of the posts at +-110 mm, and the handle came up with the gripper (the base drifts ~1 cm meanwhile).
        # Grasped is what a policy is scored on; a demonstration must also have lifted the handle.
        held = float(joints()[5]) > 0.05 and off_axis < 0.03 and slide < 0.10
        ok = held and lifted > 0.5 * LIFT
        print(f"[collect] ep {ep:3d} base {int(base_ids[i]):3d}: {'OK  ' if ok else 'HELD' if held else 'FAIL'} bar {off_axis * 1000:5.1f} mm off the gripper centre, "
              f"slid {slide * 1000:4.0f} mm, handle lifted {lifted * 1000:5.1f} mm, jaw at {float(joints()[5]):.2f} rad", flush=True)
        if not ok and not conn and not replay:
            print(f"[collect]        off the target after each phase: {', '.join(left)}; joints (deg) "
                  f"{np.round(np.degrees(joints().numpy()), 0)}", flush=True)
        if args.video and cams:
            import subprocess

            import imageio_ffmpeg  # its bundled ffmpeg
            frames = [np.hstack([rec[f"image_{n}"][j] for n in image_names]) for j in range(len(rec["q"]))]
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{frames[0].shape[1]}x{frames[0].shape[0]}", "-r", str(round(1 / (DT * RECORD_EVERY))), "-i", "-",
                            "-pix_fmt", "yuv420p", str(Path(args.video) / f"ep{ep:04d}_{'ok' if held else 'fail'}.mp4")],
                           input=b"".join(np.ascontiguousarray(x).tobytes() for x in frames), check=True)
            # the episode's signals next to the video (no images), to see when a grasp was made or lost
            np.savez_compressed(Path(args.video) / f"ep{ep:04d}_{'ok' if held else 'fail'}.npz", success=held, lifted=ok, base_index=int(base_ids[i]),
                                fps=1.0 / (DT * RECORD_EVERY), **{k_: np.stack(v) for k_, v in rec.items() if v and not k_.startswith("image_")})
        if args.out and ok and not replay:  # failed attempts are not demonstrations
            np.savez_compressed(Path(args.out) / f"ep{ep:04d}.npz", **{k_: np.stack(v) for k_, v in rec.items() if v},
                                success=ok, base_index=int(base_ids[i]), phase_names=np.array([n for n, _ in PHASES]),
                                fps=1.0 / (DT * RECORD_EVERY), drone_state_names=np.array(DRONE_STATE_NAMES), joint_names=np.array([dof_names[int(j)] for j in all_idx]))
        return held, ok

    for d in (args.out, args.video):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
    oks = []
    for ep in range(args.episodes):
        oks.append(run_episode(ep))
        if args.gui:  # cameras moved by hand in the viewport: report where they are now
            for n, c in cams.items():
                prim = stage.GetPrimAtPath(c.prim_path)
                print(f"[collect] {n}_cam at {prim.GetPath()}: translate {tuple(round(v, 4) for v in prim.GetAttribute('xformOp:translate').Get())}, "
                      f"orient (w, x, y, z) {prim.GetAttribute('xformOp:orient').Get()}", flush=True)
    print(f"[collect] success {sum(h for h, _ in oks)}/{len(oks)} grasped, {sum(o for _, o in oks)}/{len(oks)} grasped and lifted; jitter {args.jitter * 100:.1f} cm, action noise {args.action_noise} rad", flush=True)
    while args.gui and simulation_app.is_running():  # keep the window open after the last episode
        sim.render()
    print("BUILD done")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
