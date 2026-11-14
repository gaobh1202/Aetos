"""Collect SO101 grasp demonstrations from random drone-base poses (arm only, real contact).

The drone is bolted to a moving mount instead of flying: each episode the mount is put at one
feasible base pose (scripts/arm_planning/feasible_base_set.py) and then drifts a little, the way
a hovering drone does. The arm starts in its default posture and a scripted expert drives the
gripper centre -- targets are poses in the world, re-solved every step by the Cartesian
controller -- to a pregrasp, onto the handle bar, closes the jaw, and lifts the handle. All
contacts are on; an episode counts (and is saved) only if the bar is still in the jaw after the lift.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python scripts/collect_arm_grasp_demo.py --episodes 20 --out outputs/arm_grasp_demos

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
p.add_argument("--gui", action="store_true")
args = p.parse_args()

from aetos_multiam import init_simulation_app  # noqa: E402

GRASP_LOCAL = (0.0108, -0.0004, -0.0682)  # gripper centre on the 'gripper' link; = aerial_transport_track.GRASP_LOCAL
HANDLE_IN_BOX = (-0.42, 0.0, 0.34)  # bar centre of handle 0 (CarryBox.handle_local_offsets)
TABLE_Z = 0.75
JAW_OPEN, JAW_KP, JAW_KD = 1.5, 10.0, 0.5  # the authored jaw drive (kp 0.35) cannot hold anything
RETREAT, LIFT = 0.08, 0.04  # m: pregrasp distance back along the fingers; how far the handle is lifted
PHASES = (("reach", 2.0), ("approach", 1.5), ("close", 1.0), ("lift", 1.5), ("hold", 1.0))
DT, RECORD_EVERY = 0.01, 4

simulation_app = init_simulation_app({"headless": not args.gui})
try:
    import torch
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.core.prims import RigidPrimView
    from omni.isaac.core.simulation_context import SimulationContext
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    from scipy.spatial.transform import Rotation as R

    import omni_drones.utils.kit as kit_utils

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
        from omni.isaac.sensor import Camera

        gripper_prim = [x for x in Usd.PrimRange(stage.GetPrimAtPath(root))
                        if x.GetName() == "gripper" and x.HasAPI(UsdPhysics.RigidBodyAPI)][0].GetPath().pathString
        # name: (parent prim, position, orientation wxyz), in the parent link's frame.
        for name, (parent, eye, quat) in {
            # Out beside the fixed finger, clear of the gripper body, looking at the gripper centre.
            "wrist": (gripper_prim, (-0.10, 0.0, -0.01), look_at((-0.10, 0.0, -0.01), np.asarray(GRASP_LOCAL), (0.0, 0.0, 1.0))),
            # Under the airframe ahead of the arm, looking down and slightly forward, turned ~45 deg
            # about its axis (set by hand in the viewport; Euler xyz, deg, as the Property panel shows it).
            "base": (f"{root}/base_link", (0.17458, 0.00309, -0.05512),
                     np.roll(R.from_euler("xyz", (1.857, 9.042, 44.59), degrees=True).as_quat(), 1)),
        }.items():
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

    def joints():
        return view.get_joint_positions().reshape(-1, view.num_dof)[0, all_idx]

    def handle_pose():
        bp, bq = box_view.get_world_poses()
        return T(bp[0].numpy(), bq[0].numpy()) @ T(HANDLE_IN_BOX)

    def ee_pose():
        pos, quat = ee.pose()
        return T(pos[0].numpy(), quat[0].numpy())

    def run_episode(ep):
        i = int(rng.integers(len(bases)))
        box_world = T((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), TABLE_Z + 0.002),
                      np.roll(R.from_euler("z", rng.uniform(-20, 20), degrees=True).as_quat(), 1))
        base = box_world @ T(HANDLE_IN_BOX) @ bases[i]
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
        t, noise = 0.0, torch.zeros(5)
        rec = {k: [] for k in ("q", "action", "action_executed", "phase", "base_pose", "box_pose", "ee_pose",
                               "ee_target", *(f"image_{n}" for n in cams))}

        def step(target, phase, k):
            nonlocal q_cmd, t
            t += DT
            drift = (amp * np.sin(2 * np.pi * freq * t + phase0)).sum(0)
            m = base @ T(drift[:3], np.roll(R.from_euler("z", drift[3]).as_quat(), 1))
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
            if args.out and phase >= 0 and k % RECORD_EVERY == 0:
                for n, c in cams.items():
                    rec[f"image_{n}"].append(c.get_rgba()[:, :, :3].copy())
                bpos, bquat = box_view.get_world_poses()
                for key, v in (("q", joints().numpy()), ("action", clean.numpy()), ("action_executed", sent.numpy()),
                               ("phase", phase), ("base_pose", m), ("box_pose", T(bpos[0].numpy(), bquat[0].numpy())),
                               ("ee_pose", ee_pose()), ("ee_target", target if target is not None else ee_pose())):
                    rec[key].append(np.asarray(v, dtype=np.float32 if key != "phase" else np.int64))
            sim.step(render=bool(cams) and bool(args.out) and (k + 1) % RECORD_EVERY == 0 or args.gui)

        for k in range(50):  # settle in the default posture
            step(None, -1, k)
        h0 = handle_pose()
        grasp = h0 @ grasps[i]  # the gripper-centre pose this base's plan ends in, on the live handle
        pre = grasp @ T((0.0, 0.0, RETREAT))
        start = ee_pose()
        k, left = 0, []
        for phase, (name, seconds) in enumerate(PHASES):
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
        ok = float(joints()[5]) > 0.05 and off_axis < 0.03 and slide < 0.10 and lifted > 0.5 * LIFT
        print(f"[collect] ep {ep:3d} base {i:3d}: {'OK  ' if ok else 'FAIL'} bar {off_axis * 1000:5.1f} mm off the gripper centre, "
              f"slid {slide * 1000:4.0f} mm, handle lifted {lifted * 1000:5.1f} mm, jaw at {float(joints()[5]):.2f} rad", flush=True)
        if not ok:
            print(f"[collect]        off the target after each phase: {', '.join(left)}; joints (deg) "
                  f"{np.round(np.degrees(joints().numpy()), 0)}", flush=True)
        if args.out and ok:  # failed attempts are not demonstrations
            np.savez_compressed(Path(args.out) / f"ep{ep:04d}.npz", **{k_: np.stack(v) for k_, v in rec.items() if v},
                                success=ok, base_index=i, phase_names=np.array([n for n, _ in PHASES]),
                                fps=1.0 / (DT * RECORD_EVERY), joint_names=np.array([dof_names[int(j)] for j in all_idx]))
        return ok

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    oks = []
    for ep in range(args.episodes):
        oks.append(run_episode(ep))
        if args.gui:  # cameras moved by hand in the viewport: report where they are now
            for n, c in cams.items():
                prim = stage.GetPrimAtPath(c.prim_path)
                print(f"[collect] {n}_cam at {prim.GetPath()}: translate {tuple(round(v, 4) for v in prim.GetAttribute('xformOp:translate').Get())}, "
                      f"orient (w, x, y, z) {prim.GetAttribute('xformOp:orient').Get()}", flush=True)
    print(f"[collect] success {sum(oks)}/{len(oks)}; jitter {args.jitter * 100:.1f} cm, action noise {args.action_noise} rad", flush=True)
    while args.gui and simulation_app.is_running():  # keep the window open after the last episode
        sim.render()
    print("BUILD done")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
