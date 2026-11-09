"""Assert that AerialAssembly's two gears can actually be assembled.

No robot involved: teleport each gear onto its own shaft, let it settle, then
spin one and check the other turns. That is the task's hard requirement --
"both gears go in, and they mesh" -- and it depends on asset geometry and on
SDF collision being active, neither of which any smoke test would catch.

    python scripts/check_gear_assembly.py headless=true

Exits non-zero with the failing measurement if anything is off.
"""
import logging
import math
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

import hydra
import torch
from omegaconf import OmegaConf

SETTLE_STEPS = 400
SPIN_STEPS = 300
SPIN_RATE = 4.0           # rad/s held on the driven gear
# Drop height above the seated pose. Has to stay under 20 mm: a gear's bore
# starts 5 mm above its body origin and the shafts are 25 mm tall, so releasing
# any higher than that leaves the bore completely clear of the shaft and the
# gear just topples onto the plate.
SEAT_CLEARANCE_BASE_M = 0.006   # multiplied by task.gear_scale


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "AerialAssembly"
    cfg.task.env.num_envs = 1

    from aetos_multiam import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    from aetos_multiam.environments import make

    env = make(cfg.task.name, cfg, headless=cfg.headless)
    env.reset()

    base_pos = env.gear_base.get_world_pose()[0].reshape(-1, 3)
    seated = base_pos.clone()
    drop = seated.clone()
    drop[:, 2] += SEAT_CLEARANCE_BASE_M * env.gear_scale
    ident = torch.zeros(1, 4, device=env.device)
    ident[:, 0] = 1.0

    # One at a time, the way two robots would do it: dropping both together
    # jams them tooth-on-tooth before either reaches its shaft.
    for gear in env.gears:
        gear._view.set_world_poses(drop, ident)
        # A slow spin about the shaft axis lets the teeth phase themselves
        # against the neighbour already in place instead of jamming on it.
        v = torch.zeros(1, 6, device=env.device)
        # Scaled down with the gear: the same 2 rad/s that nudges a 42 mm gear
        # into phase throws a 168 mm one straight off its shaft.
        v[:, 5] = 2.0 / env.gear_scale
        gear._view.set_velocities(v)
        for _ in range(SETTLE_STEPS):
            env.sim.step(render=False)

    ok = True
    seated_z = []
    for gear in env.gears:
        p, q = gear._view.get_world_poses()
        p = p.reshape(-1, 3)
        dz = float(p[0, 2] - seated[0, 2])
        dxy = float(torch.norm(p[0, :2] - seated[0, :2]))
        tilt = 2.0 * math.degrees(math.acos(min(1.0, abs(float(q.reshape(-1, 4)[0, 0])))))
        seated_z.append(dz)
        logging.info("%s: dz=%+.4f m  dxy=%.4f m  tilt=%.2f deg", gear.name, dz, dxy, tilt)
        # Seated means it slid down the shaft and stayed on it, upright.
        tol = env.gear_scale
        if not (-0.001 * tol <= dz <= 0.004 * tol and dxy < 0.004 * tol and tilt < 10.0):
            logging.error("%s did NOT seat on its shaft", gear.name)
            ok = False

    # Mesh test: spin one gear up once and let it coast. If the teeth are
    # interleaved the other one picks up rotation the opposite way. Driving a
    # velocity every step instead does not work at larger gear_scale -- the
    # contact solver cancels the command and both gears just sit there.
    driven, follower = env.gears[0], env.gears[1]
    spin = torch.zeros(1, 6, device=env.device)
    spin[:, 5] = SPIN_RATE
    driven._view.set_velocities(spin)
    peak_follower = 0.0
    peak_sign = 0.0
    for _ in range(SPIN_STEPS):
        env.sim.step(render=False)
        wd = float(driven._view.get_velocities().reshape(-1, 6)[0, 5])
        wf = float(follower._view.get_velocities().reshape(-1, 6)[0, 5])
        if abs(wf) > peak_follower:
            peak_follower, peak_sign = abs(wf), wd * wf
    logging.info("driven %s spun up to %+.2f rad/s; follower %s peaked at %.3f rad/s "
                 "(%s direction)", driven.name, SPIN_RATE, follower.name, peak_follower,
                 "opposite" if peak_sign < 0 else "same")
    if not (peak_follower > 0.05 and peak_sign < 0.0):
        logging.error("gears are NOT meshing -- follower peak %.3f rad/s, sign product %+.3f",
                      peak_follower, peak_sign)
        ok = False

    if ok:
        logging.info("Gear assembly check PASSED: both gears seated and meshing.")
    else:
        logging.error("Gear assembly check FAILED -- see the measurements above.")
    env.close()
    simulation_app.close()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
