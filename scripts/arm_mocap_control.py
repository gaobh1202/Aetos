"""End-effector Cartesian control of a mounted arm by a mocap target (drone base pinned in place).

The red box is the mocap: the arm's tool centre (for a gripper: its centre; for the dexterous airbot: the hand flange)
follows its position exactly, and its orientation as far as the arm's joints allow.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python scripts/arm_mocap_control.py                 # GUI: drag / rotate /World/mocap in the viewport
    python scripts/arm_mocap_control.py --test          # headless: mocap moves +-4 cm along x, y, z
    python scripts/arm_mocap_control.py --test --video outputs/arm_cartesian/soarm_mocap.mp4
"""
import argparse
import math
import traceback

p = argparse.ArgumentParser()
p.add_argument("--preset", default="iris650_soarm100")
p.add_argument("--test", action="store_true", help="headless scripted mocap path, prints the tracking error")
p.add_argument("--video", default=None, help="with --test: film it to this mp4")
p.add_argument("--position-only", action="store_true", help="ignore the mocap's orientation")
args = p.parse_args()

from aetos_multiam import init_simulation_app  # noqa: E402

# Per preset: tool-centre link and point in that link's frame, a bent start posture (deg), and
# drive gains that differ from the USD's. Presets not listed track their eef link's origin.
ARMS = {
    "iris650_soarm100": dict(
        link="gripper", point=(0.0108, -0.0004, -0.0682),  # between the jaws; = aerial_transport_track.GRASP_LOCAL
        q0=[0.0, -82.4, 36.6, 45.4, 11.0], gains={"shoulder_pan": (24.0, 3.4)}),  # as task.shoulder_pan_gains
    # Centre of the two finger pads (robot.xml: pads span z -0.015..0.025 on link6, closing along y).
    "amov950_airbot_bottom": dict(link="link6", point=(0.0, 0.0, 0.005), q0=[0.0, -45.0, 90.0, 0.0, 45.0, 0.0]),
    # Centre of the flange the Inspire hand bolts to: the hand base's origin, (0, 0.0848, 0) on link5
    # (inspire_hand.py). The hand's mount joint is the wrist roll, so it is controlled with the arm.
    "amov950_airbot_dexterous_bottom": dict(link="r_base_link", extra_joints=["gripper_MountJoint"],
                                            q0=[0.0, -45.0, 90.0, 0.0, 45.0, 0.0],
                                            residual=0.001),  # the hand's finger drives diverge above this
    # Gripper centres below: middle of the fingers' bounding box, measured in the tool link's frame.
    "uaquad600_gripper": dict(link="ee_link", point=(0.0111, -0.0321, -0.0746), pose=True),
    # Three pitch joints and a roll: the gripper centre only moves in the arm's plane.
    "fahexa750_gripper": dict(link="ee_link", point=(0.134, 0.0, 0.0), pose=True, planar=True),
    # Two parallel joints: position in the arm's plane only. Point: on the gripper axis, mid-finger.
    "hummingbird340_2dof_gripper": dict(link="arm_carrier/tool_adapter", point=(0.0, 0.0, 0.08), q0=[-45.0, 90.0],
                                        planar=True, step=0.02),
}

simulation_app = init_simulation_app({"headless": args.test})
try:
    import torch
    from omni.isaac.core.objects import VisualCuboid
    from omni.isaac.core.simulation_context import SimulationContext
    from pxr import Gf, UsdLux, UsdPhysics

    import omni_drones.utils.kit as kit_utils

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
    UsdLux.DomeLight.Define(sim.stage, "/World/dome").CreateIntensityAttr(1500.0)
    am = AerialManipulatorBase(get_preset(args.preset), name="arm_test")
    BASE = (0.0, 0.0, 1.5)
    root = am.spawn(translations=[BASE])[0].GetPath().pathString
    # Pin the drone to the world. Kept outside the articulation so it stays floating-base for
    # PhysX and the Jacobian keeps the layout EndEffector reads.
    pin = UsdPhysics.FixedJoint.Define(sim.stage, "/World/pin")
    pin.CreateBody1Rel().SetTargets([f"{root}/base_link"])
    pin.CreateLocalPos0Attr(Gf.Vec3f(*BASE))
    pin.CreateExcludeFromArticulationAttr(True)
    sim.reset()
    am.initialize()
    view = am.drone._view
    cfg = ARMS.get(args.preset, {})
    if cfg.get("residual", 1.0) != 1.0:
        freeze_arm_dynamics(am, residual=cfg["residual"])

    dof_names = list(view._dof_names)
    fingers = set(am.drone_spec.bundled_gripper_joint_names)
    arm_idx = torch.tensor([int(i) for i in am.arm_joint_indices if dof_names[int(i)] not in fingers]
                           + [dof_names.index(j) for j in cfg.get("extra_joints", [])])
    for name, (kp, kd) in cfg.get("gains", {}).items():
        view.set_gains(kps=torch.full((1, 1), kp), kds=torch.full((1, 1), kd),
                       joint_indices=torch.tensor([dof_names.index(name)]))
    link = cfg.get("link") or (am.manipulator_spec.eef_body_name if am.manipulator_spec is not None else "link6")
    ee = EndEffector(view, link, arm_idx, cfg.get("point", (0.0, 0.0, 0.0)))
    lim = view.get_dof_limits().reshape(-1, view.num_dof, 2)[0][arm_idx]
    q_lo, q_hi = lim[:, 0], lim[:, 1]
    full_pose = cfg.get("pose", len(arm_idx) >= 5) and not args.position_only
    ctl = CartesianController(damping=0.02, position_first=len(arm_idx) < 6)

    def q_now():
        return view.get_joint_positions().reshape(-1, view.num_dof)[:, arm_idx]

    def base_pos():
        return view.get_world_poses()[0].reshape(-1, 3)[0]

    q_cmd = (torch.deg2rad(torch.tensor([cfg["q0"]])) if "q0" in cfg else q_now() + 0.4).clamp(q_lo + 0.05, q_hi - 0.05)
    hold_idx = torch.tensor([int(i) for i in am.teleop_joint_indices if int(i) not in arm_idx.tolist()], dtype=torch.long)
    hold_q = view.get_joint_positions().reshape(-1, view.num_dof)[:, hold_idx].clone()

    # Presets whose drives are unusable (arm_write="teleport", see presets.py) get joint positions written directly.
    write = view.set_joint_positions if am.cfg.arm_write == "teleport" else view.set_joint_position_targets

    def step(render):
        if len(hold_idx):  # gripper / hand joints stay where they are
            write(hold_q.reshape(view.count, 1, -1), joint_indices=hold_idx)
        write(q_cmd.reshape(view.count, 1, -1), joint_indices=arm_idx)
        if am.cfg.arm_write == "teleport":
            # A written position keeps the joint's velocity: the hand's undriven wrist-roll mount
            # joint was spinning at ~900 deg/s under it, 9 deg off by the next step.
            view.set_joint_velocities(torch.zeros(view.count, 1, view.num_dof))
        sim.step(render=render)

    for _ in range(300):  # reach the start posture
        step(False)
    base0 = base_pos().clone()
    home_p, home_q = (x.clone() for x in ee.pose())
    mocap = VisualCuboid("/World/mocap", position=home_p[0], orientation=home_q[0],
                         scale=torch.tensor([0.03, 0.02, 0.01]), color=torch.tensor([1.0, 0.1, 0.1]))

    def plane_normal():
        """World direction the tool centre cannot move in (a planar arm's plane normal)."""
        return torch.linalg.svd(ee.jacobian()[0, :3])[0][:, -1]

    def track():
        """One control step toward the mocap's current pose; returns (position, orientation) error."""
        global q_cmd
        m_p, m_q = (x.reshape(1, -1).float() for x in mocap.get_world_pose())
        pos, quat = ee.pose()
        e_p, e_r = pose_error(m_p, m_q, pos, quat)
        J = ee.jacobian()
        if cfg.get("planar"):  # follow the mocap within the arm's plane, ignore how far it is off it
            n = plane_normal()
            e_p = e_p - n * (e_p * n).sum(-1, keepdim=True)
        q_cmd = ctl.step(J, e_p, e_r if full_pose else None, q_now(), q_cmd, q_lo, q_hi, DT)
        return e_p.norm().item(), e_r.norm().item()

    print(f"[mocap] {args.preset}: tool centre on '{link}' at {cfg.get('point', (0, 0, 0))}, "
          f"{'pose' if full_pose else 'position'} tracking, joints {[dof_names[int(i)] for i in arm_idx]}", flush=True)

    if not args.test:
        import omni.usd

        omni.usd.get_context().get_selection().set_selected_prim_paths(["/World/mocap"], True)
        print("[mocap] drag or rotate /World/mocap (the red box) in the viewport; close the window to quit", flush=True)
        while simulation_app.is_running():
            track()
            step(True)
    else:
        frames_dir, n_frame = None, 0
        if args.video:
            import shutil
            from pathlib import Path

            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            vp = get_active_viewport()
            cs = ViewportCameraState("/OmniverseKit_Persp", vp)
            c = home_p[0]
            r = max(0.45, 1.3 * float((c - base0).norm()))  # frame bigger arms from further away
            cs.set_position_world(Gf.Vec3d(float(c[0]) + r, float(c[1]) - r, float(c[2]) + 0.45 * r), True)
            cs.set_target_world(Gf.Vec3d(*[float(v) for v in (2 * c + base0) / 3]), True)
            for _ in range(20):
                sim.render()
            frames_dir = Path(args.video).with_suffix("")
            shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True)

        def run(goal, seconds, k0=[0]):
            """Move the mocap in a straight line to ``goal`` over ``seconds``; errors seen on the way."""
            global n_frame
            start = mocap.get_world_pose()[0].reshape(1, 3).float()
            out = []
            n = int(seconds / DT)
            for k in range(n):
                mocap.set_world_pose(position=(start + (goal - start) * (k + 1) / n)[0])
                out.append(track())
                step(False)
                k0[0] += 1
                if frames_dir is not None and k0[0] % 4 == 0:  # 25 Hz of simulated time
                    sim.render()
                    capture_viewport_to_file(vp, str(frames_dir / f"{n_frame:05d}.png"))
                    sim.render()
                    n_frame += 1
            return out

        SPEED, D = 0.04, cfg.get("step", 0.04)  # m/s, m
        ok = True
        for axis in range(3):
            for sign in (1.0, -1.0):
                if cfg.get("planar") and abs(float(plane_normal()[axis])) > 0.7:
                    print(f"[mocap]   {'+' if sign > 0 else '-'}{'xyz'[axis]}: out of the arm's plane, not followed", flush=True)
                    continue
                goal = home_p.clone()
                goal[0, axis] += sign * D
                moving = run(goal, D / SPEED)
                held = run(goal, 1.0)[-30:]
                run(home_p, D / SPEED)
                run(home_p, 0.5)
                lag = max(e for e, _ in moving)
                e_pos, e_ori = (sum(x) / len(x) for x in zip(*held))
                at_limit = bool((torch.minimum(q_cmd - q_lo, q_hi - q_cmd) < 0.01).any())
                ok &= e_pos < 0.003 or at_limit
                print(f"[mocap]   {'+' if sign > 0 else '-'}{'xyz'[axis]} {D * 100:.0f} cm at {SPEED * 100:.0f} cm/s: "
                      f"lag while moving {lag * 1000:4.1f} mm, held {e_pos * 1000:4.1f} mm"
                      f"{f' / {math.degrees(e_ori):4.1f} deg' if full_pose else ''}"
                      f"{' -- at a joint limit' if at_limit else ''}", flush=True)
        drift = (base_pos() - base0).norm().item()
        ok &= drift < 1e-3
        print(f"[mocap] base moved {drift * 1000:.2f} mm", flush=True)
        print("RESULT:", "OK" if ok else "FAIL", flush=True)
        if args.video:
            import subprocess

            import imageio_ffmpeg  # its bundled ffmpeg; a system build may lack the png decoder

            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "25",
                            "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.video], check=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"[mocap] video: {args.video}", flush=True)
    print("BUILD done")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
