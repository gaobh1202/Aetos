"""Regression check for AerialTransportTrack (the CTBR drone-controller env).

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python debug/check_aerial_transport_track.py [--num-envs 16] [--speed 16,64,256]

A. replay the demo's own actions open-loop from the exact demo state (reset
   noise off): tracking must start near zero -- proves the reference, the
   dt/substep alignment and the reset are consistent with the recording.
B. the demo's closed-loop CTBR expert (Lee position loop + PX4 attitude P) on
   the demo's setpoints, reset noise on: must track for the whole episode
   without terminating -- proves a good policy exists in this action space.
C. uniform random actions: must terminate early.
Then control-step throughput for each --speed num_envs (fresh process each).
Prints RESULT: OK / FAIL.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=16)
p.add_argument("--speed", type=str, default="16,64,256")
p.add_argument("--speed-only", action="store_true", help=argparse.SUPPRESS)
args = p.parse_args()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=[
        "task=AerialTransportTrack", "headless=true", f"++task.env.num_envs={args.num_envs}"])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)

ok = True
try:
    import torch
    from torchrl.envs.utils import step_mdp

    from aetos_multiam.controllers import ctbr_from_lee, make_position_controller, normalize_ctbr
    from aetos_multiam.environments import make as make_env
    from aetos_multiam.objects import CarryBox

    env = make_env("AerialTransportTrack", cfg, headless=True)
    N, dev = env.num_envs, env.device

    def run(policy, steps, label):
        td = env.reset()
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        first_fail = torch.full((N,), -1, device=dev)
        errs, rews = [], []
        for k in range(steps):
            td.set(("agents", "action"), policy(td))
            out = env.step(td)
            nxt = out["next"]
            failed = nxt["terminated"].reshape(N) & alive
            first_fail[failed] = k
            alive &= ~nxt["done"].reshape(N)
            errs.append(env._pos_err[alive].mean().item() if alive.any() else float("nan"))
            rews.append(nxt["agents", "reward"].reshape(N, -1)[alive].mean().item() if alive.any() else float("nan"))
            td = step_mdp(out)
            if not alive.any():
                break
        n_fail = int((first_fail >= 0).sum())
        e = torch.tensor(errs)
        print(f"[check] {label}: steps={len(errs)} pos_err@1/10/50/end="
              f"{[round(float(e[i]), 4) for i in (0, min(9, len(e) - 1), min(49, len(e) - 1), len(e) - 1)]} m  "
              f"mean_reward/agent/step={torch.tensor(rews).nanmean().item():.3f}  failed_envs={n_fail}/{N} "
              f"median_first_fail={int(first_fail[first_fail >= 0].median()) if n_fail else None}", flush=True)
        return e, n_fail

    if not args.speed_only:
        # A. open-loop demo actions, no reset noise.
        env.reset_pos_noise, env.reset_rot_noise = 0.0, 0.0
        eA, _ = run(lambda td: env._ref("action"), 60, "A open-loop demo actions")
        if not float(eA[0]) < 0.01:
            print("[check] FAIL A: first-step tracking error not ~0 -> reference/reset misaligned", flush=True)
            ok = False

        # B. closed-loop expert, noise on.
        env.reset_pos_noise, env.reset_rot_noise = 0.02, __import__("math").radians(2.0)
        lees = [make_position_controller(am.drone, 9.81).to(dev) for am in env.aerial_manipulators]
        max_thr = env.rate_controllers[0].max_total_thrust

        def expert(td):
            acts = []
            for i, (am, lee) in enumerate(zip(env.aerial_manipulators, lees)):
                st = am.drone.get_state()[:, 0, :13]
                rate, thr = ctbr_from_lee(lee, st, env._ref("sp_pos")[:, i], env._ref("sp_vel")[:, i],
                                          env._ref("yaw")[:, i:i + 1], max_thr)
                # The demo expert had the payload in its mass feedforward.
                thr = thr + (env._ref("load") * env.payload_scale * CarryBox.mass_kg * 9.81 / 2 / max_thr).unsqueeze(-1)
                acts.append(normalize_ctbr(rate, thr))
            return torch.stack(acts, 1)

        eB, nB = run(expert, cfg.task.env.max_episode_length, "B closed-loop CTBR expert")
        if nB > 0 or float(eB.nanmean()) > 0.1:
            print("[check] FAIL B: the expert itself cannot track this env", flush=True)
            ok = False

        # C. random actions.
        _, nC = run(lambda td: torch.rand(N, 2, 4, device=dev) * 2 - 1, 150, "C random actions")
        if nC < N // 2:
            print("[check] FAIL C: random actions should mostly terminate", flush=True)
            ok = False
    else:
        td = env.reset()
        td.set(("agents", "action"), env._ref("action"))
        for _ in range(20):
            td = step_mdp(env.step(td))
            td.set(("agents", "action"), env._ref("action"))
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(100):
            td = step_mdp(env.step(td))
            td.set(("agents", "action"), env._ref("action"))
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"[speed] num_envs={N}: {100 / dt:.1f} control steps/s, {100 * N / dt:.0f} env-steps/s, "
              f"{100 * N * 2 / dt:.0f} agent-steps/s, GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    ok = False
finally:
    if not args.speed_only:
        print(f"RESULT: {'OK' if ok else 'FAIL'}", flush=True)
    app.close()

if not args.speed_only and args.speed:
    for n in [int(x) for x in args.speed.split(",") if x]:
        subprocess.run([sys.executable, "-u", __file__, "--num-envs", str(n), "--speed-only"])
