"""Record iris650_soarm100 flying takeoff -> forward -> left -> yaw -> forward.

Writes a PNG sequence to --out-dir; encode it with e.g.
    ffmpeg -y -framerate 25 -i <out-dir>/frame_%05d.png \
           -c:v libx264 -pix_fmt yuv420p -crf 20 out.mp4
(the system ffmpeg here has no libx264 -- use the one imageio_ffmpeg ships:
 python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").
Capture is every 4th physics step at 100 Hz, so 25 fps plays back real-time.

Driven exactly the way examples/aerial_manipulator_control_demo.py's --mode dynamic
drives a drone: a keyboard-style {vx, vy, vz, yaw_rate} intent per phase
integrates into a moving target_pos/target_yaw setpoint (vx/vy rotated by the
current target yaw, so "forward" tracks where the drone is facing), and
LeePositionController turns that setpoint into per-rotor thrusts. All motion
comes out of rotor forces -- the lean is the controller's own geometric law,
not scripted.
"""
import argparse
import math
import os
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", required=True, help="Directory for the PNG sequence.")
parser.add_argument("--capture-every", type=int, default=4)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

from aetos_multiam import init_simulation_app  # noqa: E402

simulation_app = init_simulation_app({"headless": True})

OUT_DIR = args.out_dir
CAPTURE_EVERY = args.capture_every
DT = 1 / 100

# (label, steps, vx_body, vy_body, vz, yaw_rate)
PHASES = [
    ("settle",   100, 0.0, 0.0, 0.0, 0.0),
    ("takeoff",  200, 0.0, 0.0, 0.8, 0.0),
    ("hold",      80, 0.0, 0.0, 0.0, 0.0),
    ("forward",  250, 1.2, 0.0, 0.0, 0.0),
    ("hold",     100, 0.0, 0.0, 0.0, 0.0),
    ("left",     200, 0.0, 1.2, 0.0, 0.0),
    ("hold",     100, 0.0, 0.0, 0.0, 0.0),
    ("yaw",      225, 0.0, 0.0, 0.0, 0.7),
    ("hold",     100, 0.0, 0.0, 0.0, 0.0),
    ("forward2", 250, 1.2, 0.0, 0.0, 0.0),
    ("hold",     150, 0.0, 0.0, 0.0, 0.0),
]

try:
    import torch
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf
    import omni.isaac.core.utils.prims as prim_utils

    import omni_drones.utils.kit as kit_utils
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torch import quat_axis

    from aetos_multiam.robots.presets import get_preset
    from aetos_multiam.robots.robot_base import AerialManipulatorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT, backend="torch",
        physics_prim_path="/World/PhysicsScene", device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, -9.81)},
    )
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    prim_utils.create_prim("/World/Light/Dome", "DomeLight",
                           attributes={"inputs:intensity": 1500.0})
    prim_utils.create_prim("/World/Light/Sphere", "SphereLight", translation=(2.0, 2.0, 6.0),
                           attributes={"inputs:intensity": 150000.0, "inputs:radius": 0.4})
    # Static ground markers -- a chase camera over a bare plane makes a moving
    # drone look parked. Visual-only prims (no collider), so they cannot
    # interfere with the flight being recorded.
    for ix in range(-2, 8):
        for iy in range(-4, 5):
            prim_utils.create_prim(
                f"/World/Markers/m_{ix + 2}_{iy + 4}", "Cube",
                translation=(ix * 1.5, iy * 1.5, 0.1),
                attributes={"size": 0.2},
            )

    am = AerialManipulatorBase(get_preset("iris650_soarm100"), name="iris_am")
    am.spawn(translations=[(0.0, 0.0, 0.6)])
    sim.reset()
    am.initialize()
    drone = am.drone
    controller = LeePositionController(9.81, drone.params).to("cpu")

    env_ids = torch.tensor([0])
    arm_targets = torch.zeros(1, drone.num_arm_joints)
    drone.reset_arm_joints(env_ids, arm_targets)

    target_pos = drone.get_state()[..., :3].reshape(1, 3).clone()
    target_yaw = torch.zeros(1)

    vp = get_active_viewport()
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)

    frame_idx = 0
    step = 0
    for label, n, vx, vy, vz, yaw_rate in PHASES:
        print(f"[rec] phase '{label}' {n} steps  step={step} frame={frame_idx}", flush=True)
        for _ in range(n):
            # keyboard-style intent -> moving world setpoint
            yaw = float(target_yaw[0])
            target_pos[:, 0] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT
            target_pos[:, 1] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
            target_pos[:, 2] += vz * DT
            target_yaw += yaw_rate * DT

            state = drone.get_state()[..., :13].squeeze(0)
            action = controller.compute(state, target_pos=target_pos, target_yaw=target_yaw)
            drone.apply_action(action)
            drone.hold_arm_joints(arm_targets, zero_velocity=True)
            sim.step(render=False)

            if step % CAPTURE_EVERY == 0:
                s = drone.get_state()
                p = s[..., :3].reshape(-1)
                px, py, pz = float(p[0]), float(p[1]), float(p[2])
                # World-fixed viewing DIRECTION (not body-relative), so the yaw
                # phase is actually visible instead of being cancelled out.
                cam_state.set_position_world(Gf.Vec3d(px - 2.0, py - 2.2, pz + 1.1), True)
                cam_state.set_target_world(Gf.Vec3d(px, py, pz), True)
                sim.render()
                capture_viewport_to_file(vp, f"{OUT_DIR}/frame_{frame_idx:05d}.png")
                frame_idx += 1
            step += 1

    s = drone.get_state()
    p = s[..., :3].reshape(-1)
    up = quat_axis(s[..., 3:7].reshape(1, 4), axis=2).reshape(-1)
    print(f"[rec] done: {frame_idx} frames, final pos=({float(p[0]):+.2f},"
          f"{float(p[1]):+.2f},{float(p[2]):+.2f}) yaw_cmd={float(target_yaw[0]):.3f} rad "
          f"up_z={float(up[2]):.4f}", flush=True)
    print("RESULT: OK" if float(up[2]) > 0.99 else "RESULT: FAIL", flush=True)
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
