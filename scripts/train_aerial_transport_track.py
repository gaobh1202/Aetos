"""SAC on AerialTransportTrack: one shared CTBR policy for both drones.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python scripts/train_aerial_transport_track.py --num-envs 32 --steps 20000

Self-contained on purpose: OmniDrones' SACPolicy predates its current policy
interface and only starts learning once a 1M buffer is full. Each agent's
(obs, action, reward, next obs) is one transition; the replay buffer is the
part a TD-MPC-style learner can reuse later.
Writes log.csv and actor.pt to --out.

Evaluate a trained actor on whole demos (deterministic actions, from t = 0):
    python scripts/train_aerial_transport_track.py --eval outputs/aerial_transport_track/sac/actor.pt \
        --num-envs 8 --video outputs/aerial_transport_track/sac/eval.mp4
"""
import argparse
import copy
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
p = argparse.ArgumentParser()
p.add_argument("--task", default="AerialTransportTrack", help="or AerialGraspAssist (training only; --eval is demo-tracking specific)")
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--steps", type=int, default=20000, help="control steps (each = num_envs * 2 transitions)")
p.add_argument("--out", type=str, default=str(REPO / "outputs/aerial_transport_track/sac"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--warmup", type=int, default=5000, help="random-action transitions before learning")
p.add_argument("--updates-per-step", type=int, default=8)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--buffer", type=int, default=1_000_000)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--log-every", type=int, default=200)
p.add_argument("--init", type=str, default=None,
               help="continue from ckpt.pt (actor + critics) or actor.pt (critics start fresh: see --critic-warmup)")
p.add_argument("--critic-warmup", type=int, default=1500,
               help="with an actor-only --init: control steps of critic-only updates before the actor moves")
p.add_argument("--eval", type=str, default=None, help="actor.pt to evaluate instead of training")
p.add_argument("--video", type=str, default=None, help="with --eval: mp4 of the env nearest the origin")
p.add_argument("--video-every", type=int, default=4)
p.add_argument("--override", nargs="*", default=[], help="extra hydra overrides, e.g. ++task.arm_dynamics_residual=0.5")
args = p.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=[
        f"task={args.task}", "headless=true", f"++task.env.num_envs={args.num_envs}", f"seed={args.seed}",
        # Eval: a huge horizon forces every episode to start at demo t = 0 and run to its end.
        *(["++task.min_episode_horizon=1000000", "++task.env.max_episode_length=100000"] if args.eval else []),
        *(["++task.visuals=true", "++task.warehouse=true"] if args.video else []), *args.override])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)

try:
    import torch
    from torch import nn

    from aetos_multiam.environments import make as make_env

    torch.manual_seed(args.seed)
    env = make_env(args.task, cfg, headless=True)
    N, dev = env.num_envs, env.device
    OBS, ACT = ("agents", "observation"), ("agents", "action")
    td = env.reset()
    obs_dim, act_dim, n_agents = td[OBS].shape[-1], 4, td[OBS].shape[1]

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, 256), nn.LayerNorm(256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o)).to(dev)

    actor = mlp(obs_dim, 2 * act_dim)
    q1, q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
    log_alpha = torch.tensor(-1.6, device=dev, requires_grad=True)  # alpha0 ~ 0.2
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_q = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args.lr)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)

    def act(obs, deterministic=False):
        mu, log_std = actor(obs).chunk(2, -1)
        log_std = log_std.clamp(-5, 2)
        u = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.9189385).sum(-1, keepdim=True) \
            - torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, logp

    # Replay buffer on the GPU: flat per-agent transitions.
    B = args.buffer
    buf = dict(o=torch.zeros(B, obs_dim, device=dev), a=torch.zeros(B, act_dim, device=dev),
               r=torch.zeros(B, 1, device=dev), o2=torch.zeros(B, obs_dim, device=dev),
               d=torch.zeros(B, 1, device=dev))
    ptr, size = 0, 0

    actor_frozen_until = 0
    if args.init:
        ck = torch.load(args.init, map_location=dev)
        if "actor" in ck:
            for m, k in ((actor, "actor"), (q1, "q1"), (q2, "q2"), (q1_t, "q1_t"), (q2_t, "q2_t")):
                m.load_state_dict(ck[k])
            log_alpha.data.fill_(float(ck["log_alpha"]))
        else:  # actor only: let the fresh critics catch up before they are allowed to move it
            actor.load_state_dict(ck)
            actor_frozen_until = args.critic_warmup
            # A converged policy sits at alpha ~0.03; restarting at 0.2 re-noises it and it collapses.
            log_alpha.data.fill_(-3.5)

    def update(train_actor=True):
        idx = torch.randint(0, size, (args.batch,), device=dev)
        o, a, r, o2, d = (buf[k][idx] for k in ("o", "a", "r", "o2", "d"))
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = act(o2)
            x2 = torch.cat([o2, a2], -1)
            target = r + args.gamma * (1 - d) * (torch.min(q1_t(x2), q2_t(x2)) - alpha * logp2)
        x = torch.cat([o, a], -1)
        loss_q = (q1(x) - target).pow(2).mean() + (q2(x) - target).pow(2).mean()
        opt_q.zero_grad(); loss_q.backward(); opt_q.step()
        if not train_actor:
            return loss_q.item(), float("nan")
        a_new, logp = act(o)
        xn = torch.cat([o, a_new], -1)
        loss_a = (alpha * logp - torch.min(q1(xn), q2(xn))).mean()
        opt_a.zero_grad(); loss_a.backward(); opt_a.step()
        loss_alpha = -(log_alpha * (logp.detach() - act_dim).mean())  # target entropy = -act_dim
        opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()
        with torch.no_grad():
            for t, s in ((q1_t, q1), (q2_t, q2)):
                for pt, ps in zip(t.parameters(), s.parameters()):
                    pt.lerp_(ps, 0.005)
        return loss_q.item(), -logp.mean().item()

    if args.eval:
        import shutil
        import subprocess

        actor.load_state_dict(torch.load(args.eval, map_location=dev))
        frames_dir = None
        if args.video:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            from pxr import Gf

            cam_env = int(env.central_env_idx)
            o = env.envs_positions[cam_env].tolist()
            vp = get_active_viewport()
            cs = ViewportCameraState("/OmniverseKit_Persp", vp)
            cs.set_position_world(Gf.Vec3d(o[0] + 2.5, o[1] - 5.5, 3.2), True)
            cs.set_target_world(Gf.Vec3d(o[0], o[1] + 1.0, 0.7), True)
            for _ in range(20):
                env.sim.render()  # render() pauses physics; simulation_app.update() would step it
            frames_dir = Path(args.video).with_suffix("")
            shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True)
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        failed = torch.zeros(N, dtype=torch.bool, device=dev)
        steps = torch.zeros(N, device=dev)
        arm_err = torch.zeros(N, device=dev)  # worst arm joint deviation from the demo, rad
        err_sum, err_max, ret = torch.zeros(N, device=dev), torch.zeros(N, device=dev), torch.zeros(N, device=dev)
        demo_len = env.demo["len"][env.demo_idx].clone()
        demo_id = env.demo_idx.clone()
        k = frame = 0
        while alive.any():
            with torch.no_grad():
                td.set(ACT, act(td[OBS], deterministic=True)[0])
            out, td = env.step_and_maybe_reset(td)
            nxt = out["next"]
            e = env._pos_err.reshape(N, -1).max(-1).values  # worse of the two drones
            aq = (nxt[OBS][..., -6:-1] - env._ref("arm_q")[..., :5]).abs().reshape(N, -1).max(-1).values
            arm_err = torch.where(alive, torch.maximum(arm_err, aq), arm_err)
            steps += alive
            err_sum += e * alive
            err_max = torch.where(alive, torch.maximum(err_max, e), err_max)
            ret += nxt["agents", "reward"].reshape(N, -1).mean(-1) * alive
            failed |= nxt["terminated"].reshape(N) & alive
            alive &= ~nxt["done"].reshape(N)
            k += 1
            # Film only the first episode of the camera env.
            if frames_dir is not None and alive[cam_env] and k % args.video_every == 0:
                env.sim.render()
                capture_viewport_to_file(vp, str(frames_dir / f"{frame:05d}.png"))
                env.sim.render()
                frame += 1
        for i in range(N):
            print(f"[eval] env {i}: demo {int(demo_id[i])} steps {int(steps[i])}/{int(demo_len[i])} "
                  f"{'FAILED' if failed[i] else 'ok'} return {ret[i]:.0f} "
                  f"pos_err mean {err_sum[i] / steps[i] * 100:.1f} cm max {err_max[i] * 100:.1f} cm", flush=True)
        print(f"[eval] {int((~failed).sum())}/{N} episodes completed, mean pos_err {(err_sum / steps).mean() * 100:.1f} cm, "
              f"worst {err_max.max() * 100:.1f} cm, mean return {ret.mean():.0f}, "
              f"worst arm joint deviation {arm_err.max().rad2deg():.1f} deg", flush=True)
        if frames_dir is not None and frame:
            import imageio_ffmpeg  # its bundled ffmpeg; a system build may lack the png decoder

            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate",
                            str(round(1 / (cfg.task.sim.dt * cfg.task.sim.substeps * args.video_every))),
                            "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.video], check=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"[eval] video: {args.video} ({frame} frames)", flush=True)
        raise SystemExit

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.csv", "w")
    log.write("step,transitions,elapsed_s,episodes,return,episode_len,pos_error_m,failed_frac,ref_progress,q_loss,entropy,alpha\n")
    ep = {k: [] for k in ("return", "episode_len", "pos_error", "terminated_early", "ref_progress")}
    t0, lq, ent = time.time(), float("nan"), float("nan")
    for step in range(1, args.steps + 1):
        obs = td[OBS]
        with torch.no_grad():
            a = torch.rand(N, n_agents, act_dim, device=dev) * 2 - 1 if size < args.warmup and not args.init else act(obs)[0]
        td.set(ACT, a)
        out, td = env.step_and_maybe_reset(td)
        nxt = out["next"]
        n = N * n_agents
        idx = (ptr + torch.arange(n, device=dev)) % B
        buf["o"][idx], buf["a"][idx] = obs.reshape(n, -1), a.reshape(n, -1)
        buf["r"][idx], buf["o2"][idx] = nxt["agents", "reward"].reshape(n, 1), nxt[OBS].reshape(n, -1)
        # Bootstrap through time-outs; only real failures end the return.
        buf["d"][idx] = nxt["terminated"].reshape(N, 1, 1).expand(N, n_agents, 1).reshape(n, 1).float()
        ptr, size = (ptr + n) % B, min(size + n, B)
        done = nxt["done"].reshape(N)
        if done.any():
            for k in ep:  # the env's own "terminated_early" stat lags one step, so take it from the done flags
                src = nxt["terminated"].float() if k == "terminated_early" else nxt["stats", k]
                ep[k] += src.reshape(N)[done].tolist()
        if size >= args.warmup:
            for _ in range(args.updates_per_step):
                lq, ent = update(step > actor_frozen_until)
        if step % args.log_every == 0 or step == args.steps:
            m = {k: (sum(v) / len(v) if v else float("nan")) for k, v in ep.items()}
            row = (step, step * n, time.time() - t0, len(ep["return"]), m["return"], m["episode_len"],
                   m["pos_error"], m["terminated_early"], m["ref_progress"], lq, ent, log_alpha.exp().item())
            log.write(",".join(f"{v:.4g}" for v in row) + "\n"); log.flush()
            print(f"[train] step {step} ({step * n} transitions, {step / (time.time() - t0):.1f} steps/s) "
                  f"episodes={len(ep['return'])} return={m['return']:.1f} len={m['episode_len']:.0f} "
                  f"pos_err={m['pos_error']:.3f} m failed={m['terminated_early']:.2f} "
                  f"q_loss={lq:.3g} entropy={ent:.2f} alpha={log_alpha.exp().item():.3f}", flush=True)
            ep = {k: [] for k in ep}
            torch.save(actor.state_dict(), out_dir / "actor.pt")
            torch.save({"actor": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
                        "q1_t": q1_t.state_dict(), "q2_t": q2_t.state_dict(), "log_alpha": log_alpha.item()},
                       out_dir / "ckpt.pt")
except SystemExit:
    pass
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
