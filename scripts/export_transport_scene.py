"""Write what the planner sees of an AerialTransport episode: scene.json (names, poses) and scene.png.

    python scripts/export_transport_scene.py --out outputs/iris650_soarm100_AerialTransport_planning/ep0 --seed 0

The box is placed as scripts/collect_aerial_transport_demo.py places it (same ranges); the image is an
overview of the bay from above and behind the robots.
"""
import argparse
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
p = argparse.ArgumentParser()
p.add_argument("--out", default=str(REPO / "outputs/iris650_soarm100_AerialTransport_planning/ep0"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--starts", type=float, nargs="+", default=None, metavar="X Y Z",
               help="start position of each robot (three numbers per robot); default: the painted zones' centres")
p.add_argument("--yaw-offset-deg", type=float, nargs="+", default=None,
               help="with --starts: each robot heads at the box (arm and gripper forward), turned by this much")
p.add_argument("--instruction", default="Move the box from the pick table onto the green rectangle on the drop table.")
args = p.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=["task=AerialTransport", "headless=true", "++task.env.num_envs=1"])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)

try:
    import numpy as np
    import torch
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf

    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
    from aetos_multiam.environments import make as make_env
    from aetos_multiam.environments.collaborative.aerial_transport import DROP_TABLE_Y_M, PICK_TABLE_Y_M, TABLE_TOP_Z_M

    rng = np.random.default_rng(args.seed)
    env = make_env("AerialTransport", cfg, headless=True)
    env.reset()
    dev, origin = env.device, env.envs_positions[0].cpu().numpy()
    box = env.carry_box._view
    bxy, byaw = rng.uniform(-0.10, 0.10, 2), math.radians(rng.uniform(-15.0, 15.0))
    bpos = np.array([bxy[0], PICK_TABLE_Y_M + bxy[1], TABLE_TOP_Z_M + 0.002])
    box.set_world_poses(torch.as_tensor(bpos + origin, dtype=torch.float32, device=dev).reshape(1, 3),
                        euler_to_quaternion(torch.tensor([[0.0, 0.0, byaw]], device=dev)))
    box.set_velocities(torch.zeros(1, 6, device=dev))
    for _ in range(10):
        env.sim.step(False)
    c, s = math.cos(byaw), math.sin(byaw)
    parts = []
    for i, (hx, hy, hz) in enumerate(env.carry_box.handle_local_offsets()):
        out = np.sign(hx) * np.array([c, s, 0.0])
        parts.append(dict(name=f"handle_{i}", position=np.round(bpos + np.array([c * hx - s * hy, s * hx + c * hy, hz]), 3).tolist(),
                          outward=np.round(out, 3).tolist()))
    if args.starts:
        for i, am in enumerate(env.aerial_manipulators):
            pos = np.array(args.starts[3 * i: 3 * i + 3])
            yaw = math.atan2(bpos[1] - pos[1], bpos[0] - pos[0]) + math.radians((args.yaw_offset_deg or [0.0] * 8)[i])
            am.drone.set_world_poses(torch.as_tensor(pos + origin, dtype=torch.float32, device=dev).reshape(1, 3),
                                     euler_to_quaternion(torch.tensor([[0.0, 0.0, yaw]], device=dev)))
            am.drone.set_velocities(torch.zeros(1, 6, device=dev))
        env.sim.step(False)
    robots = []
    for i, am in enumerate(env.aerial_manipulators):
        st = am.drone.get_state()[0, 0]
        robots.append(dict(name=f"am_{i}", type="quadrotor (0.65 m) with a 5-DoF arm and parallel gripper",
                           position=[round(float(v), 3) for v in st[:3]],
                           yaw_deg=round(math.degrees(float(quaternion_to_euler(st[3:7])[2])), 1)))
    scene = dict(task="AerialTransport", instruction=args.instruction, robots=robots, objects=[
        dict(name="carry_box", description="box with one handle bar at each end; too wide for one robot",
             position=np.round(bpos, 3).tolist(), yaw_deg=round(math.degrees(byaw), 1), mass_kg=1.0, parts=parts),
        dict(name="pick_table", position=[0.0, PICK_TABLE_Y_M, TABLE_TOP_Z_M], yaw_deg=0.0),
        dict(name="drop_zone", description="green rectangle on the drop table: where the box has to stand",
             position=[0.0, DROP_TABLE_Y_M, TABLE_TOP_Z_M + 0.002], yaw_deg=0.0)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    vp = get_active_viewport()
    cs = ViewportCameraState("/OmniverseKit_Persp", vp)
    cs.set_position_world(Gf.Vec3d(float(origin[0]), float(origin[1]) - 7.0, 7.5), True)
    cs.set_target_world(Gf.Vec3d(float(origin[0]), float(origin[1]) + 0.9, 0.5), True)
    for _ in range(30):
        env.sim.render()
    capture_viewport_to_file(vp, str(out / "scene.png"))
    for _ in range(10):
        env.sim.render()
    print(f"[scene] {out}/scene.json, scene.png", flush=True)
    print(json.dumps(scene, indent=1), flush=True)
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
