"""Feasible drone-base poses for the SO101 grasping the carry-box handle.

Given the object (handle frame: origin at the bar centre, box axes, bar along y) and the
hand-object grasp pose, sample drone ``base_link`` poses (x, y, z, yaw, small tilt) around the
nominal one and keep those that pass, in order:

1. drone influence -- the rotor discs stay ``--clearance`` away from the box and the bar, and the
   drone sits diagonally above the handle (elevation within ``--elevation``), not level with it
   and not straight overhead;
2. reach -- cuRobo plans the arm collision-free from its default posture to the grasp (the bar is
   a cylinder: rotation about it is free, and the grasp may slide ``--bar-slide`` along it);
3. comfort -- at the grasp every joint is at least ``--limit-margin`` from its limits.

    conda activate armplan
    python scripts/arm_planning/feasible_base_set.py --num 2000

Everything saved is in the handle frame, so it applies wherever the box is. Joint angles are in
the simulator's convention (see so101_common.py).
"""
import argparse
import os.path as osp
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from so101_common import ROOT, T_DRONE_ARMBASE, T

DATA = osp.join(ROOT, "aetos_multiam", "data", "aerial_transport")
# iris_650: rotor hubs in the base_link frame (assets/usd/iris_650.yaml), the outer pair's y used for all four.
ROTORS = np.array([[sx * 0.171, sy * 0.29, 0.03] for sx in (1, -1) for sy in (1, -1)])
BOX_HALF = np.array([0.45, 0.20, 0.11])  # CarryBox body; its centre is at BOX_IN_HANDLE
BOX_IN_HANDLE = np.array([0.42, 0.0, -0.23])
BAR_HALF = 0.13  # the bar runs along the handle frame's y
Q_LIMITS = np.radians([[-110, 110], [-100, 100], [-100, 90], [-95, 95], [-160, 160]])  # the simulated arm's


def rotor_clearance(drone_in_handle, prop_radius):
    """Smallest distance (m) from any rotor disc to the box body or the handle bar."""
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rim = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1) * prop_radius
    pts = (ROTORS[:, None] + np.concatenate([rim, np.zeros((1, 3))])[None]).reshape(-1, 3)
    pts = pts @ drone_in_handle[:3, :3].T + drone_in_handle[:3, 3]
    to_box = np.linalg.norm(np.maximum(np.abs(pts - BOX_IN_HANDLE) - BOX_HALF, 0), axis=1)
    on_bar = np.stack([np.zeros(len(pts)), np.clip(pts[:, 1], -BAR_HALF, BAR_HALF), np.zeros(len(pts))], -1)
    return float(min(to_box.min(), np.linalg.norm(pts - on_bar, axis=1).min()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--pos-range", type=float, nargs=3, default=[0.25, 0.25, 0.25], help="+-m about the nominal base, x y z")
    p.add_argument("--yaw-range", type=float, default=45.0, help="+-deg")
    p.add_argument("--tilt-range", type=float, default=3.0, help="+-deg of roll and pitch (a hovering drone is near level)")
    p.add_argument("--prop-radius", type=float, default=0.19, help="m; 15 inch props assumed for the 650 frame")
    p.add_argument("--clearance", type=float, default=0.10, help="m between a rotor disc and the box / bar")
    p.add_argument("--elevation", type=float, nargs=2, default=[20.0, 80.0], help="deg of the drone above the handle")
    p.add_argument("--bar-slide", type=float, default=0.08, help="m the grasp may sit from the bar centre")
    p.add_argument("--limit-margin", type=float, default=6.0, help="deg to the nearest joint limit at the grasp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=osp.join(DATA, "so101_feasible_bases.npz"))
    args = p.parse_args()

    from so101_planner import GraspPlanner  # cuRobo: slow to import

    nominal = np.load(osp.join(DATA, "so101_default_pose.npz"))
    drone0, q_default = nominal["drone_in_handle"], nominal["q_default"]
    gp = GraspPlanner(free_bar_rotation=True, bar_slide=args.bar_slide)
    rng = np.random.default_rng(args.seed)

    bases, why, q_grasp, trajs, dts, margin, clear, elev, grasps = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for _ in range(args.num):
        xyz = rng.uniform(-1, 1, 3) * args.pos_range
        rpy = np.radians(rng.uniform(-1, 1, 3) * [args.tilt_range, args.tilt_range, args.yaw_range])
        base = drone0.copy()  # offsets along the handle frame's axes, rotation about the drone's own
        base[:3, 3] += xyz
        base[:3, :3] = base[:3, :3] @ R.from_euler("xyz", rpy).as_matrix()
        c = rotor_clearance(base, args.prop_radius)
        e = np.degrees(np.arctan2(base[2, 3], np.hypot(base[0, 3], base[1, 3])))
        bases.append(base), clear.append(c), elev.append(e)
        m, q, tr, dt = np.nan, np.full(5, np.nan), None, np.nan
        if c < args.clearance:
            reason = "rotor"
        elif not args.elevation[0] <= e <= args.elevation[1]:
            reason = "elevation"
        else:
            res = gp.plan(gp.handle @ base @ T_DRONE_ARMBASE, q_default)
            if res is None:
                reason = "reach"
            else:
                tr, dt = res
                q = tr[-1]
                m = np.degrees(np.minimum(q - Q_LIMITS[:, 0], Q_LIMITS[:, 1] - q).min())
                reason = "ok" if m >= args.limit_margin else "limit"
        why.append(reason), q_grasp.append(q), margin.append(m), trajs.append(tr), dts.append(dt)
        # Where this base's plan actually puts the grasp frame (it may turn about the bar and slide along it).
        grasps.append(base @ T_DRONE_ARMBASE @ gp.fk(q) if tr is not None else np.full((4, 4), np.nan))

    bases, why = np.array(bases), np.array(why)
    ok = why == "ok"
    planned = sum(w not in ("rotor", "elevation") for w in why)
    print(f"\n{args.num} bases sampled +-{args.pos_range} m, yaw +-{args.yaw_range} deg about the nominal; "
          f"{time.time() - t0:.0f} s ({planned} planned)")
    for w, label in (("rotor", f"rotor disc within {args.clearance * 100:.0f} cm of the box / bar"),
                     ("elevation", f"drone not {args.elevation[0]:.0f}-{args.elevation[1]:.0f} deg above the handle"),
                     ("reach", "arm cannot reach the grasp"),
                     ("limit", f"a joint within {args.limit_margin:.0f} deg of its limit"), ("ok", "feasible")):
        print(f"  {(why == w).sum():5d}  {label}")
    if ok.any():
        off = bases[:, :3, 3] - drone0[:3, 3]
        yaw = np.degrees([R.from_matrix(drone0[:3, :3].T @ b[:3, :3]).as_euler("xyz")[2] for b in bases])
        print("feasible fraction by offset from the nominal base (handle frame: x away from the box, z up):")
        for name, v, scale, unit in (("x", off[:, 0], 100, "cm"), ("y", off[:, 1], 100, "cm"),
                                     ("z", off[:, 2], 100, "cm"), ("yaw", yaw, 1, "deg")):
            edges = np.linspace(v.min(), v.max() + 1e-9, 6)
            cells = [f"{lo * scale:+.0f}..{hi * scale:+.0f}: {ok[(v >= lo) & (v < hi)].mean() * 100:3.0f}%"
                     for lo, hi in zip(edges[:-1], edges[1:])]
            print(f"  {name:3s} ({unit})  " + "   ".join(cells))
        steps = max(len(t) for t in trajs if t is not None)
        np.savez_compressed(
            args.out, drone_in_handle=bases, reason=why, feasible=ok, q_grasp=np.array(q_grasp),
            traj=np.stack([np.pad(t, ((0, steps - len(t)), (0, 0)), mode="edge") for t, o in zip(trajs, ok) if o]).astype(np.float32),
            traj_len=np.array([len(t) for t, o in zip(trajs, ok) if o]), dt=np.array(dts)[ok],
            limit_margin_deg=np.array(margin), rotor_clearance_m=np.array(clear), elevation_deg=np.array(elev),
            q_default=q_default, drone_in_handle_nominal=drone0, grasp_in_handle=gp.handle_grasp,
            grasp_in_handle_each=np.array(grasps),
            args=np.str_(str(vars(args))))
        print("saved", args.out, "(traj / traj_len / dt cover the feasible bases only, in order)")
    gp.p.destroy()


if __name__ == "__main__":  # cuRobo spawns a helper process that re-imports this file
    assert 2.0 < rotor_clearance(T((-0.5, 0, 2.0), (1, 0, 0, 0)), 0.19) < 2.2  # 2 m above the bar
    assert rotor_clearance(T((0.42, 0, -0.2), (1, 0, 0, 0)), 0.19) == 0.0  # hub inside the box
    main()
