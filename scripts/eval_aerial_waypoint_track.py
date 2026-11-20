"""Evaluate a drone controller in AerialWaypointTrack: the classical cascaded controller, or a trained actor.

    python scripts/eval_aerial_waypoint_track.py --num-envs 256                      # classical baseline
    python scripts/eval_aerial_waypoint_track.py --actor outputs/iris650_soarm100_AerialWaypointTrack/sac/actor.pt

Every env runs whole episodes (10 s each) until --episodes have finished; results are split by what the
arm was doing (dataset replay / random walk / static pose).

Film one episode instead (the camera follows the reference point, drawn as a red ball):
    python scripts/eval_aerial_waypoint_track.py --actor .../actor.pt --num-envs 4 --video out.mp4
The same --seed gives every controller the same route, arm motion and payload.
"""
import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
p = argparse.ArgumentParser()
p.add_argument("--actor", default=None, help="actor.pt from train_aerial_transport_track.py (omit: classical controller)")
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--episodes", type=int, default=512)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--video", default=None, help="mp4 of the first episode of --video-env (then stop)")
p.add_argument("--video-env", type=int, default=0)
p.add_argument("--video-every", type=int, default=2, help="control steps per frame")
p.add_argument("--override", nargs="*", default=[], help="extra hydra overrides, e.g. ++task.arm_obs=state")
args = p.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=["task=AerialWaypointTrack", "headless=true", f"seed={args.seed}",
                                                  f"++task.env.num_envs={args.num_envs}",
                                                  *(["++task.visuals=true"] if args.video else []), *args.override])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)

try:
    import torch
    from torch import nn

    from aetos_multiam.environments import make as make_env
    from aetos_multiam.environments.single.aerial_waypoint_track import ARM_MODES

    torch.manual_seed(args.seed)
    env = make_env("AerialWaypointTrack", cfg, headless=True)
    N, dev = env.num_envs, env.device
    OBS, ACT = ("agents", "observation"), ("agents", "action")
    td = env.reset()
    actor = None
    if args.actor:
        obs_dim = td[OBS].shape[-1]
        actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 8)).to(dev)  # as the training script's
        actor.load_state_dict(torch.load(args.actor, map_location=dev))
    if args.video:
        import shutil
        import subprocess

        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf, UsdGeom
        import omni.usd

        ve = args.video_env
        origin = env.envs_positions[ve]
        ball = UsdGeom.Sphere.Define(omni.usd.get_context().get_stage(), "/World/ref_marker")
        ball.CreateRadiusAttr(0.03)
        ball.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.1, 0.1)])
        ball_pos = UsdGeom.Xformable(ball).AddTranslateOp()
        vp = get_active_viewport()
        cs = ViewportCameraState("/OmniverseKit_Persp", vp)

        def aim():
            c = (env._at(env.ref_pos)[ve] + origin).tolist()
            ball_pos.Set(Gf.Vec3d(*c))
            cs.set_position_world(Gf.Vec3d(c[0] + 1.6, c[1] - 1.6, c[2] + 0.7), True)
            cs.set_target_world(Gf.Vec3d(c[0], c[1], c[2] - 0.1), True)

        aim()
        for _ in range(20):
            env.sim.render()  # render() pauses physics; simulation_app.update() would step it
        frames_dir = Path(args.video).with_suffix("")
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True)
        frame = k = 0
        while True:
            with torch.no_grad():
                a = env.expert_action() if actor is None else torch.tanh(actor(td[OBS]).chunk(2, -1)[0])
            td.set(ACT, a)
            mode, load = ARM_MODES[int(env.arm_mode[ve])], float(env.load_n[ve])
            out, td = env.step_and_maybe_reset(td)
            nxt = out["next"]
            if nxt["done"][ve].any():
                st = nxt["stats"]
                print(f"[video] {'classical controller' if actor is None else args.actor}: arm {mode}, payload {load:.1f} N, "
                      f"{'FAILED' if nxt['terminated'][ve].any() else 'completed'} after {int(st['episode_len'][ve])} steps, "
                      f"pos_err {float(st['pos_error'][ve]) * 100:.1f} cm, tilt {float(st['tilt_deg'][ve]):.2f} deg", flush=True)
                break
            k += 1
            if k % args.video_every == 0:
                aim()
                env.sim.render()
                capture_viewport_to_file(vp, str(frames_dir / f"{frame:05d}.png"))
                env.sim.render()
                frame += 1
        import imageio_ffmpeg  # its bundled ffmpeg; a system build may lack the png decoder

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate",
                        str(round(1 / (env.ctrl_dt * args.video_every))), "-i", str(frames_dir / "%05d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.video], check=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[video] {args.video} ({frame} frames)", flush=True)
        raise SystemExit
    keys = ("pos_error", "tilt_deg", "yaw_error_deg", "terminated_early", "episode_len", "return")
    ep, modes = {k: [] for k in keys}, []
    while len(modes) < args.episodes:
        with torch.no_grad():
            a = env.expert_action() if actor is None else torch.tanh(actor(td[OBS]).chunk(2, -1)[0])
        td.set(ACT, a)
        mode = env.arm_mode.clone()
        out, td = env.step_and_maybe_reset(td)
        nxt = out["next"]
        done = nxt["done"].reshape(N)
        if done.any():
            for k in keys:
                src = nxt["terminated"].float() if k == "terminated_early" else nxt["stats", k]
                ep[k] += src.reshape(N)[done].tolist()
            modes += mode[done].tolist()
    m = torch.tensor(modes)
    ep = {k: torch.tensor(v) for k, v in ep.items()}
    print(f"[eval] {'classical controller' if actor is None else args.actor}, {len(m)} episodes", flush=True)
    for name, s in [*((n, m == i) for i, n in enumerate(ARM_MODES)), ("all", torch.ones_like(m, dtype=torch.bool))]:
        if s.any():
            print(f"[eval] arm {name:8s} episodes {int(s.sum()):4d}  failed {ep['terminated_early'][s].mean() * 100:5.1f}%  "
                  f"pos_err {ep['pos_error'][s].mean() * 100:5.1f} cm  tilt {ep['tilt_deg'][s].mean():5.2f} deg  "
                  f"yaw_err {ep['yaw_error_deg'][s].mean():5.2f} deg  return {ep['return'][s].mean():6.0f}", flush=True)
except SystemExit:
    pass
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
