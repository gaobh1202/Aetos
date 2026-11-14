"""End-effector Cartesian control check, one preset at a time (hovering drone, real arm drives).

The drone hovers under its position controller; the arm bends away from its straight home pose,
then the end-effector is commanded a few cm (4 by default) along +-x, +-y, +-z (world) from where it is, keeping
its orientation when the arm has five or more joints. Prints the error left after each move.

A move is counted against the arm only if the arm can make it: arms with few joints cannot move
their end-effector in every direction (a planar arm cannot move sideways), and those moves are
reported as "outside the arm's motion" -- the drone has to supply them.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python debug/check_arm_cartesian_control.py --preset uaquad600_gripper
"""
import argparse
import math
import traceback

p = argparse.ArgumentParser()
p.add_argument("--preset", required=True)
p.add_argument("--step-cm", type=float, default=None, help="default: START[preset], else 4")
p.add_argument("--q0", type=float, nargs="*", default=None, help="start posture, deg (default: START[preset], "
                                                                  "else 0.4 rad on every joint)")
p.add_argument("--link", default=None, help="end-effector link (default: the manipulator spec's eef body)")
p.add_argument("--position-only", action="store_true", help="do not hold the orientation")
p.add_argument("--residual", type=float, default=None, help="arm dynamics residual (default: the preset's)")
p.add_argument("--video", default=None, help="film the moves to this mp4 (2x speed; a red ball marks the target)")
args = p.parse_args()

from aetos_multiam import init_simulation_app  # noqa: E402

# A working posture per preset to start from (deg), away from the straight-arm singularity and
# clear of the airframe; plus the end-effector link and any drive gains that differ from the USD's.
START = {
    "iris650_soarm100": dict(q0=[0.0, -82.4, 36.6, 45.4, 11.0], link="gripper",  # so101_default_pose.npz
                             gains={"shoulder_pan": (24.0, 3.4)},  # as task.shoulder_pan_gains
                             # With the arm's dynamics frozen (the preset's 0.1) its soft drives let
                             # a downward move swing slowly around the target; unfrozen it settles.
                             residual=1.0),
    "amov950_airbot_bottom": dict(q0=[0.0, -45.0, 90.0, 0.0, 45.0, 0.0]),
    "amov950_airbot_dexterous_bottom": dict(q0=[0.0, -45.0, 90.0, 0.0, 45.0]),
    # A short two-link arm: bent elbow (straight is singular) and a step that fits its reach.
    "hummingbird340_2dof_gripper": dict(q0=[-45.0, 90.0], step_cm=2.0),
}

simulation_app = init_simulation_app({"headless": True})
try:
    import torch
    from omni.isaac.core.simulation_context import SimulationContext

    import omni_drones.utils.kit as kit_utils
    from omni_drones.utils.torch import quat_axis

    from aetos_multiam.controllers import make_position_controller
    from aetos_multiam.controllers.arm import CartesianController, EndEffector, pose_error
    from aetos_multiam.robots.presets import get_preset
    from aetos_multiam.robots.robot_base import AerialManipulatorBase
    from aetos_multiam.utils import freeze_arm_dynamics

    DT = 0.01
    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT, backend="torch",
        physics_prim_path="/World/PhysicsScene", device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, -9.81)},
    )
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    am = AerialManipulatorBase(get_preset(args.preset), name="test_am")
    am.spawn(translations=[(0.0, 0.0, 2.0)])
    sim.reset()
    am.initialize()
    start = START.get(args.preset, {})
    args.step_cm = args.step_cm or start.get("step_cm", 4.0)
    residual = args.residual if args.residual is not None else start.get("residual", am.cfg.arm_dynamics_residual)
    freeze_arm_dynamics(am, residual=residual)
    drone, view = am.drone, am.drone._view
    hover = make_position_controller(drone, 9.81).to("cpu")

    dof_names = list(view._dof_names)
    fingers = set(am.drone_spec.bundled_gripper_joint_names)  # a gripper baked into the drone's USD
    arm_idx = torch.tensor([int(i) for i in am.arm_joint_indices if dof_names[int(i)] not in fingers])
    n = len(arm_idx)
    hold_ori = n >= 5 and not args.position_only
    for name, (kp, kd) in start.get("gains", {}).items():
        j = torch.tensor([dof_names.index(name)])
        view.set_gains(kps=torch.full((1, 1), kp), kds=torch.full((1, 1), kd), joint_indices=j)
    link = args.link or start.get("link") or (am.manipulator_spec.eef_body_name if am.manipulator_spec is not None else "link6")
    ee = EndEffector(view, link, arm_idx)
    lim = view.get_dof_limits().reshape(-1, view.num_dof, 2)[0][arm_idx]
    q_lo, q_hi = lim[:, 0], lim[:, 1]
    ctl = CartesianController()

    def q_now():
        return view.get_joint_positions().reshape(-1, view.num_dof)[:, arm_idx]

    pos0 = drone.get_state()[..., :3].reshape(1, 3).clone()
    yaw0 = torch.zeros(1)
    q0 = args.q0 or start.get("q0")
    q_cmd = (torch.deg2rad(torch.tensor([q0])) if q0 else q_now() + 0.4).clamp(q_lo + 0.05, q_hi - 0.05)

    # Gripper / hand joints are held where they are: left undriven, a multi-finger hand flails.
    hold_idx = torch.tensor([int(i) for i in am.teleop_joint_indices if int(i) not in arm_idx.tolist()], dtype=torch.long)
    hold_q = view.get_joint_positions().reshape(-1, view.num_dof)[:, hold_idx].clone()

    # Presets whose drives are unusable (arm_write="teleport", see presets.py) get joint positions
    # written directly: the controller's output is the same, only how it reaches the joints differs.
    write = view.set_joint_positions if am.cfg.arm_write == "teleport" else view.set_joint_position_targets

    frames_dir, n_step, n_frame = None, 0, 0

    def step():
        global n_step, n_frame
        n_step += 1
        if frames_dir is not None and n_step % 4 == 0:  # 25 Hz of simulated time
            sim.render()
            capture_viewport_to_file(vp, str(frames_dir / f"{n_frame:05d}.png"))
            sim.render()
            n_frame += 1
        if len(hold_idx):
            write(hold_q.reshape(view.count, 1, -1), joint_indices=hold_idx)
        state = drone.get_state()[..., :13].squeeze(0)
        drone.apply_action(hover.compute(state, target_pos=pos0, target_yaw=yaw0))
        write(q_cmd.reshape(view.count, 1, -1), joint_indices=arm_idx)
        sim.step(render=False)

    q_start = q_cmd.clone()

    def settle(steps):  # hover with the arm driven to the start posture
        global q_cmd
        q_cmd = q_start.clone()
        for _ in range(steps):
            step()

    settle(300)
    deg = lambda t: [round(math.degrees(float(v)), 1) for v in t.reshape(-1)]  # noqa: E731
    kps, kds = view.get_gains()
    print(f"[cart] {args.preset}: {n} arm joints {[dof_names[int(i)] for i in arm_idx]}, end-effector link '{link}', "
          f"arm dynamics residual {residual}\n"
          f"[cart]   start posture (deg) {deg(q_now())}, drives lag by {deg(q_start - q_now())}, "
          f"limits {[(round(a), round(b)) for a, b in zip(deg(q_lo), deg(q_hi))]}\n"
          f"[cart]   drive stiffness {[round(float(v), 1) for v in kps.reshape(-1)[arm_idx]]}, damping "
          f"{[round(float(v), 2) for v in kds.reshape(-1)[arm_idx]]}", flush=True)

    if args.video:
        import shutil
        from pathlib import Path

        from omni.isaac.core.objects import VisualSphere
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf, UsdLux

        UsdLux.DomeLight.Define(sim.stage, "/World/dome").CreateIntensityAttr(1500.0)
        ball = VisualSphere("/World/target", radius=0.012, color=torch.tensor([1.0, 0.1, 0.1]))
        c = ee.pose()[0][0]
        reach = max(0.4, 1.8 * float((c - pos0[0]).norm()))  # frame small arms closer
        vp = get_active_viewport()
        cs = ViewportCameraState("/OmniverseKit_Persp", vp)
        cs.set_position_world(Gf.Vec3d(float(c[0]) + reach, float(c[1]) - reach, float(c[2]) + 0.4 * reach), True)
        cs.set_target_world(Gf.Vec3d(*[float(v) for v in (c + pos0[0]) / 2]), True)
        for _ in range(20):
            sim.render()
        frames_dir = Path(args.video).with_suffix("")
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True)

    LEVER = 0.2  # m: weighs a radian of rotation against a metre of translation when asking what the arm can do
    ok, worst = True, 0.0
    for axis in range(3):
        for sign in (1.0, -1.0):
            settle(150)  # every move starts from the same posture
            p0, quat0 = (x.clone() for x in ee.pose())
            q_begin = q_now().clone()
            d = torch.zeros(1, 3)
            d[0, axis] = sign
            # Can the arm make this move at all (pure translation, orientation kept if it is held)?
            J = ee.jacobian()[0]
            J = torch.cat([J[:3], LEVER * J[3:]]) if hold_ori else J[:3]
            want = torch.cat([d[0], torch.zeros(3)]) if hold_ori else d[0]
            # Directions that need more than ~1.3 rad of joint travel for this step do not count as reachable.
            U, S, _ = torch.linalg.svd(J, full_matrices=False)
            U = U[:, S > args.step_cm / 100 / 1.3]
            miss = float((want - U @ (U.T @ want)).norm())
            can = miss < 0.3
            target = p0 + d * args.step_cm / 100
            if args.video:
                ball.set_world_pose(position=target[0])
            errs, trace = [], []
            for k in range(300):
                pos, quat = ee.pose()
                e_p, e_r = pose_error(target, quat0, pos, quat)
                q_cmd = ctl.step(ee.jacobian(), e_p, e_r if hold_ori else None, q_now(), q_cmd, q_lo, q_hi, DT)
                step()
                if k % 30 == 0:
                    trace.append(round(e_p.norm().item() * 1000))
                if k >= 270:
                    errs.append((e_p.norm().item(), e_r.norm().item()))
            e_pos, e_ori = (sum(x) / len(x) for x in zip(*errs))
            s = drone.get_state()
            tilt = math.degrees(math.acos(float(quat_axis(s[..., 3:7].reshape(1, 4), axis=2)[0, 2].clamp(-1, 1))))
            at_limit = bool((torch.minimum(q_cmd - q_lo, q_hi - q_cmd) < 0.01).any())
            counted = can and not at_limit
            if counted:
                worst = max(worst, e_pos)
            ok &= bool(torch.isfinite(s[..., :3]).all()) and tilt < 20 and (not counted or e_pos < 0.01)
            why = "" if counted else (" -- stopped at a joint limit" if can else
                                      f" -- the arm cannot make this move ({miss:.2f} of it is outside its motion)")
            print(f"[cart]   {'+' if sign > 0 else '-'}{'xyz'[axis]} {args.step_cm:.0f} cm: left {e_pos * 1000:5.1f} mm"
                  f"{f' / {math.degrees(e_ori):.1f} deg' if hold_ori else ' (position only)'}, "
                  f"drone moved {(s[..., :3].reshape(1, 3) - pos0).norm() * 100:.1f} cm, tilt {tilt:.1f} deg{why}", flush=True)
            if e_pos >= 0.01:
                print(f"[cart]     started from {deg(q_begin)}, ended at {deg(q_now())}, error every 0.3 s (mm) {trace}, moved {[round(float(v) * 1000, 1) for v in (ee.pose()[0] - p0)[0]]} mm; joint margins to limits (deg): {deg(torch.minimum(q_now() - q_lo, q_hi - q_now()))}, "
                      f"command leads joint by {deg(q_cmd - q_now())}", flush=True)
    print(f"[cart] {args.preset}: worst error on moves the arm can make {worst * 1000:.1f} mm", flush=True)
    print("RESULT:", "OK" if ok else "FAIL", flush=True)
    if args.video:
        import subprocess

        import imageio_ffmpeg  # its bundled ffmpeg; a system build may lack the png decoder

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "50",
                        "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.video], check=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[cart] video: {args.video}", flush=True)
    print("BUILD done")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
