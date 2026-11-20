"""Train the AerialWaypointTrack drone policy through a learned, differentiable world model.

    python scripts/train_aerial_waypoint_wm.py --num-envs 256 --steps 3000

The loop is SuperMimic's (Liao et al., CoRL 2026, docs/2610.28480v2.pdf), fitted to this task:

1. collect: the current policy (plus action noise) flies in the simulator; transitions go to a replay buffer;
2. fit the world model: an MLP that predicts the next PHYSICAL state (drone + arm, not a latent) from the state,
   the drone's CTBR action and what is known about the arm -- its joint target for this step, i.e. the action
   chunk the arm policy has committed to -- and the payload force. Trained on k-step autoregressive windows;
3. improve the policy: imagine H steps from buffer states through the frozen model and backpropagate the return
   through it (analytic gradients). The reward and the observation are the env's own functions, computed from
   the predicted state; a critic, fitted to GAE returns of the same imagined rollouts, closes the horizon.

Differences from the paper: the exogenous inputs (reference route, arm targets, payload) take the place of its
goal and sim context; termination is the env's rule, smoothed, instead of a learned head.

Writes log.csv, actor.pt (same network as train_aerial_transport_track.py's, so
scripts/eval_aerial_waypoint_track.py --actor evaluates it) and wm.pt to --out.
"""
import argparse
import math
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--steps", type=int, default=3000, help="control steps in the simulator (each = num_envs transitions)")
p.add_argument("--out", type=str, default=str(REPO / "outputs/iris650_soarm100_AerialWaypointTrack/wm"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--warmup", type=int, default=40, help="control steps collected before any learning")
p.add_argument("--buffer-steps", type=int, default=1000, help="replay buffer: control steps kept per env")
p.add_argument("--noise", type=float, default=0.1, help="std of the action noise during collection")
p.add_argument("--wm-window", type=int, default=8, help="k: autoregressive steps the model is fitted on")
p.add_argument("--wm-updates", type=int, default=4, help="world-model updates per control step")
p.add_argument("--wm-batch", type=int, default=2048)
p.add_argument("--wm-hidden", type=int, default=1024)
p.add_argument("--wm-no-arm", action="store_true", help="ablation: the model is not told the arm's targets")
p.add_argument("--horizon", type=int, default=32, help="H: imagined steps per rollout")
p.add_argument("--pi-updates", type=int, default=2, help="policy updates per control step")
p.add_argument("--pi-batch", type=int, default=4096, help="imagined rollouts per policy update")
p.add_argument("--critic-iters", type=int, default=8)
p.add_argument("--no-critic", action="store_true", help="ablation: no value beyond the horizon")
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--override", nargs="*", default=[], help="extra hydra overrides, e.g. ++task.arm_obs=state")
args = p.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=["task=AerialWaypointTrack", "headless=true", f"seed={args.seed}",
                                                  f"++task.env.num_envs={args.num_envs}", *args.override])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)

try:
    import torch
    from torch import nn

    from aetos_multiam.environments import make as make_env
    from aetos_multiam.environments.single import aerial_waypoint_track as awt
    from aetos_multiam.environments.single.aerial_waypoint_track import (
        EXO_DIM, S_INT, S_JP, S_JV, S_POS, S_RATE, S_ROT, S_THR, S_VEL, STATE_DIM, X_ARM, X_LOAD, policy_obs,
        tracking_errors, tracking_reward)

    torch.manual_seed(args.seed)
    env = make_env("AerialWaypointTrack", cfg, headless=True)
    N, dev = env.num_envs, env.device
    OBS, ACT, WMS = ("agents", "observation"), ("agents", "action"), ("wm", "state")
    td = env.reset()
    obs_dim, act_dim, dt = td[OBS].shape[-1], 4, env.ctrl_dt
    H, K = args.horizon, args.wm_window
    W = H + max(max(awt.LOOKAHEAD), max(awt.ARM_LOOKAHEAD)) + 1  # exogenous steps an imagined rollout reads
    rw, fail_pen = env.rw, float(env.rw.get("fail", 10.0))

    # -- networks ----------------------------------------------------------------------------------------
    def mlp(i, o, h=256):
        return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)).to(dev)

    actor = mlp(obs_dim, 2 * act_dim)  # the SAC script's actor; only the mean half is used
    with torch.no_grad():  # start at hover thrust, zero rates
        actor[-1].weight.mul_(0.01)
        actor[-1].bias.zero_()
        actor[-1].bias[3] = math.atanh(2 * env.hover_frac - 1)
    crit_dim = obs_dim + 1 + 4 + 3  # full observation + payload, throttles, rate integrator
    critic = mlp(crit_dim, 1)
    V_SCALE = 1.0 / (1.0 - args.gamma)

    def pi(obs):
        return torch.tanh(actor(obs)[..., :act_dim])

    WM_IN = 3 + 3 + 3 + 6 + 6 + 4 + 3 + 4 + 6 + 1
    WM_OUT = 3 + 3 + 3 + 3 + 6 + 6 + 4 + 3
    h = args.wm_hidden
    wm = nn.Sequential(nn.Linear(WM_IN, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                       nn.Linear(h, h), nn.SiLU(), nn.Linear(h, WM_OUT)).to(dev)
    wm_scale = torch.ones(WM_OUT, device=dev)  # std of the one-step targets; set from the first data

    def hat(v):
        z = torch.zeros_like(v[..., 0])
        return torch.stack([z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z], -1).reshape(*v.shape[:-1], 3, 3)

    def so3_exp(v):
        th2 = v.square().sum(-1)[..., None, None]
        th = (th2 + 1e-12).sqrt()
        a = torch.where(th2 < 1e-6, 1 - th2 / 6, th.sin() / th)
        b = torch.where(th2 < 1e-6, 0.5 - th2 / 24, (1 - th.cos()) / th2.clamp_min(1e-12))
        Kx = hat(v)
        return torch.eye(3, device=v.device) + a * Kx + b * Kx @ Kx

    def vee(M):  # rotation vector of a small rotation
        return 0.5 * torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)

    def split(s):
        return (s[..., S_POS], s[..., S_ROT].reshape(*s.shape[:-1], 3, 3), s[..., S_VEL], s[..., S_RATE],
                s[..., S_JP], s[..., S_JV], s[..., S_THR], s[..., S_INT])

    def wm_delta(s, s2):
        """What the model has to output to turn s into s2 (the inverse of wm_step's integration)."""
        p0, R, v, w, jp, jv, th, it = split(s)
        p2, R2, v2, w2, jp2, jv2, th2, it2 = split(s2)
        Rt = R.transpose(-1, -2)
        rot = lambda x: (Rt @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731  world -> body
        return torch.cat([rot(p2 - p0) - rot(v) * dt, vee(Rt @ R2) - w * dt, rot(v2 - v), w2 - w,
                          jp2 - jp - jv * dt, jv2 - jv, th2 - th, it2 - it], -1)

    def wm_step(s, a, arm_tgt, load):
        """One control step of the learned dynamics. Everything is expressed in the body frame, so the model
        does not depend on where the drone is or which way it faces."""
        p0, R, v, w, jp, jv, th, it = split(s)
        Rt = R.transpose(-1, -2)
        vb = (Rt @ v.unsqueeze(-1)).squeeze(-1)
        arm_in = torch.zeros_like(jp) if args.wm_no_arm else (arm_tgt - jp) / 0.3
        x = torch.cat([R[..., 2, :], vb / 2, w / 2, jp / 1.5, jv / 2, th, it / 0.5, a, arm_in, load.unsqueeze(-1) / 3], -1)
        d = wm(x) * wm_scale
        rot = lambda x: (R @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731  body -> world
        return torch.cat([p0 + rot(vb * dt + d[..., 0:3]), (R @ so3_exp(w * dt + d[..., 3:6])).reshape(*s.shape[:-1], 9),
                          v + rot(d[..., 6:9]), w + d[..., 9:12], jp + jv * dt + d[..., 12:18], jv + d[..., 18:24],
                          th + d[..., 24:28], it + d[..., 28:31]], -1)

    def state_err(s_hat, s):
        """Prediction error per component, in units of the one-step change's std."""
        p0, R, v, w, jp, jv, th, it = split(s_hat)
        p2, R2, v2, w2, jp2, jv2, th2, it2 = split(s)
        return torch.cat([p2 - p0, vee(R.transpose(-1, -2) @ R2), v2 - v, w2 - w, jp2 - jp, jv2 - jv, th2 - th, it2 - it], -1) / wm_scale

    def soft_fail(s, exo0, yaw):
        pos_err, _, _, tilt, z = tracking_errors(s, exo0, yaw)
        ok = (1 - torch.sigmoid((pos_err - env.term_pos_err) / 0.05)) * (1 - torch.sigmoid((tilt - env.term_tilt) / 0.05)) \
            * (1 - torch.sigmoid((0.3 - z) / 0.03))
        return 1 - ok

    def critic_in(s, last_a, exo, yaw):
        return torch.cat([policy_obs(s, last_a, exo, yaw, "preview"), exo[..., 0, X_LOAD:X_LOAD + 1] / 3,
                          s[..., S_THR], s[..., S_INT]], -1)

    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)

    # -- replay buffer: [control step, env] ----------------------------------------------------------------
    C = args.buffer_steps
    buf = dict(s=torch.zeros(C, N, STATE_DIM, device=dev), a=torch.zeros(C, N, act_dim, device=dev),
               pa=torch.zeros(C, N, act_dim, device=dev), s2=torch.zeros(C, N, STATE_DIM, device=dev),
               done=torch.zeros(C, N, device=dev), exo=torch.zeros(C, N, W, EXO_DIM, device=dev),
               yaw=torch.zeros(C, N, device=dev))
    ptr, size = 0, 0

    def sample(batch, span):
        """Random (step, env) whose `span` following steps are all in the buffer."""
        back = torch.randint(span, size + 1, (batch,), device=dev)
        return (ptr - back) % C, torch.randint(0, N, (batch,), device=dev)

    def fit_wm():
        t0, e = sample(args.wm_batch, K)
        steps = (t0.unsqueeze(-1) + torch.arange(K, device=dev)) % C
        ee = e.unsqueeze(-1)
        A, S2, D = buf["a"][steps, ee], buf["s2"][steps, ee], buf["done"][steps, ee]
        exo = buf["exo"][t0, e, :K]
        alive = torch.cat([torch.ones_like(D[:, :1]), (1 - D[:, :-1]).cumprod(-1)], -1)  # same episode as step 0
        s, loss = buf["s"][t0, e], 0.0
        for j in range(K):
            s = wm_step(s, A[:, j], exo[:, j, X_ARM], exo[:, j, X_LOAD])
            loss = loss + (alive[:, j] * state_err(s, S2[:, j]).square().mean(-1)).sum()
        loss = loss / alive.sum()
        opt_wm.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt_wm.step()
        return loss.item()

    @torch.no_grad()
    def wm_error():
        """Open-loop K-step error of the model on the newest data: (position mm, attitude deg) at step K."""
        t0, e = (ptr - K - torch.arange(1, 9, device=dev).repeat_interleave(N)) % C, torch.arange(N, device=dev).repeat(8)
        steps = (t0.unsqueeze(-1) + torch.arange(K, device=dev)) % C
        D = buf["done"][steps, e.unsqueeze(-1)]
        ok = D[:, :-1].sum(-1) == 0
        s, exo = buf["s"][t0, e], buf["exo"][t0, e, :K]
        for j in range(K):
            s = wm_step(s, buf["a"][steps[:, j], e], exo[:, j, X_ARM], exo[:, j, X_LOAD])
        err = state_err(s, buf["s2"][steps[:, -1], e]) * wm_scale
        return (err[ok, 0:3].norm(dim=-1).mean().item() * 1e3, err[ok, 3:6].norm(dim=-1).mean().rad2deg().item())

    def improve_policy():
        t0, e = sample(args.pi_batch, 1)
        s, prev, exo, yaw = buf["s"][t0, e], buf["pa"][t0, e], buf["exo"][t0, e], buf["yaw"][t0, e]
        for q in (*wm.parameters(), *critic.parameters()):
            q.requires_grad_(False)
        c, disc, J = torch.ones_like(yaw), 1.0, 0.0
        xs, rs, ds = [], [], []
        for t in range(H):
            xs.append(critic_in(s, prev, exo[:, t:], yaw).detach())
            a = pi(policy_obs(s, prev, exo[:, t:], yaw, env.arm_obs))
            s = wm_step(s, a, exo[:, t, X_ARM], exo[:, t, X_LOAD])
            d = soft_fail(s, exo[:, t + 1], yaw)
            r = tracking_reward(s, a, prev, exo[:, t + 1], yaw, rw) - fail_pen * d
            J = J + disc * c * r
            c, disc, prev = c * (1 - d), disc * args.gamma, a
            rs.append(r.detach()); ds.append(d.detach())
        x_last = critic_in(s, prev, exo[:, H:], yaw)
        if not args.no_critic:
            J = J + disc * c * critic(x_last).squeeze(-1) * V_SCALE
        loss = -J.mean() / H
        opt_pi.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_pi.step()
        for q in (*wm.parameters(), *critic.parameters()):
            q.requires_grad_(True)
        if not args.no_critic:  # GAE(lambda) returns of the imagined rollout, then regress the critic on them
            with torch.no_grad():
                X = torch.stack(xs)  # (H, B, crit_dim)
                Vs = torch.cat([critic(X).squeeze(-1), critic(x_last.detach()).squeeze(-1).unsqueeze(0)]) * V_SCALE
                ret, R = [None] * H, Vs[H]
                for t in reversed(range(H)):
                    R = rs[t] + args.gamma * (1 - ds[t]) * ((1 - args.lam) * Vs[t + 1] + args.lam * R)
                    ret[t] = R
                X, Y = X.reshape(-1, crit_dim), torch.stack(ret).reshape(-1) / V_SCALE
            for _ in range(args.critic_iters):
                i = torch.randint(0, len(X), (16384,), device=dev)
                lv = (critic(X[i]).squeeze(-1) - Y[i]).square().mean()
                opt_v.zero_grad(); lv.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
        return torch.stack(rs).mean().item()

    # -- the env's functions against the env itself (a wrong sign here would train the wrong task silently) ----
    exo = env.wm_exo(W)
    o_chk = policy_obs(td[WMS], env.last_action[:, 0], exo, env.ref_yaw, env.arm_obs)
    assert torch.allclose(o_chk, td[OBS][:, 0], atol=1e-5), "policy_obs does not reproduce the env's observation"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.csv", "w")
    log.write("step,transitions,elapsed_s,episodes,return,episode_len,pos_error_m,failed_frac,tilt_deg,"
              "wm_loss,wm_pos_err_mm,wm_att_err_deg,imagined_reward\n")
    keys = ("return", "episode_len", "pos_error", "terminated_early", "tilt_deg")
    ep = {k: [] for k in keys}
    t_start, l_wm, r_im, scaled = time.time(), float("nan"), float("nan"), False
    for step in range(1, args.steps + 1):
        s, obs, prev = td[WMS], td[OBS], env.last_action[:, 0].clone()
        exo, yaw = env.wm_exo(W), env.ref_yaw.clone()
        with torch.no_grad():
            a = (pi(obs[:, 0]) + args.noise * torch.randn(N, act_dim, device=dev)).clamp(-1, 1)
        td.set(ACT, a.unsqueeze(1))
        out, td = env.step_and_maybe_reset(td)
        nxt = out["next"]
        buf["s"][ptr], buf["a"][ptr], buf["pa"][ptr], buf["s2"][ptr] = s, a, prev, nxt[WMS]
        buf["done"][ptr], buf["exo"][ptr], buf["yaw"][ptr] = nxt["done"].reshape(N).float(), exo, yaw
        ptr, size = (ptr + 1) % C, min(size + 1, C)
        if step == 1:  # reward check, one real transition
            r_chk = tracking_reward(nxt[WMS], a, prev, exo[:, 1], yaw, rw) - fail_pen * nxt["terminated"].reshape(N)
            assert torch.allclose(r_chk, nxt["agents", "reward"].reshape(N), atol=1e-4), "tracking_reward != env reward"
        done = nxt["done"].reshape(N)
        if done.any():
            for k in keys:
                src = nxt["terminated"].float() if k == "terminated_early" else nxt["stats", k]
                ep[k] += src.reshape(N)[done].tolist()
        if step >= args.warmup:
            if not scaled:
                scaled = True
                wm_scale.copy_(wm_delta(buf["s"][:size], buf["s2"][:size]).reshape(-1, WM_OUT).std(0).clamp_min(1e-5))
                print("[wm] one-step target std:", [round(v, 5) for v in wm_scale.tolist()], flush=True)
                for _ in range(300):  # a first fit, so the policy's first gradients are not from a blank model
                    l_wm = fit_wm()
            for _ in range(args.wm_updates):
                l_wm = fit_wm()
            for _ in range(args.pi_updates):
                r_im = improve_policy()
        if step % args.log_every == 0 or step == args.steps:
            m = {k: (sum(v) / len(v) if v else float("nan")) for k, v in ep.items()}
            e_pos, e_att = wm_error() if scaled and size > K + 9 else (float("nan"), float("nan"))
            row = (step, step * N, time.time() - t_start, len(ep["return"]), m["return"], m["episode_len"], m["pos_error"],
                   m["terminated_early"], m["tilt_deg"], l_wm, e_pos, e_att, r_im)
            log.write(",".join(f"{v:.4g}" for v in row) + "\n"); log.flush()
            print(f"[train] step {step} ({step * N} transitions, {step / (time.time() - t_start):.2f} steps/s) "
                  f"episodes={len(ep['return'])} return={m['return']:.1f} len={m['episode_len']:.0f} "
                  f"pos_err={m['pos_error']:.3f} m failed={m['terminated_early']:.2f} tilt={m['tilt_deg']:.2f} deg | "
                  f"wm_loss={l_wm:.3g} wm_{K}step_err={e_pos:.1f} mm {e_att:.2f} deg imagined_r={r_im:.3f}", flush=True)
            ep = {k: [] for k in keys}
            torch.save(actor.state_dict(), out_dir / "actor.pt")
            torch.save({"wm": wm.state_dict(), "scale": wm_scale, "critic": critic.state_dict(), "args": vars(args)},
                       out_dir / "wm.pt")
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
