"""Plan SO101 grasps of the carry-box handle from randomly placed arm bases.

The target is the hand-object pose of the scripted demos. The arm base is displaced
from a nominal pose and cuRobo plans the arm from its default posture to that
hand-object pose. Without --default-pose the nominal pose and posture are the demo's
(arm stretched out, joints all zero); with it, the ones choose_so101_default_pose.py found.

    conda activate armplan
    python scripts/arm_planning/plan_so101_grasp.py --num 300 \\
        --default-pose aetos_multiam/data/aerial_transport/so101_default_pose.npz \\
        --out aetos_multiam/data/aerial_transport/arm_grasp_plans.npz

Saved joint angles are in the simulator's convention (see so101_common.py).
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from so101_planner import GraspPlanner

AXES = ["x", "y", "z", "roll", "pitch", "yaw"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=300)
    p.add_argument("--pos-range", type=float, default=0.12, help="base offset, +-m per axis")
    p.add_argument("--yaw-range", type=float, default=25.0, help="base yaw offset, +-deg")
    p.add_argument("--tilt-range", type=float, default=5.0, help="base roll/pitch offset, +-deg")
    p.add_argument("--strict-pose", action="store_true",
                   help="fix all 6 DoF of the hand-object pose (default: free rotation about the handle bar)")
    p.add_argument("--bar-slide", type=float, default=0.08,
                   help="the grasp point may sit this far from the bar centre along the bar, m (0: centre only)")
    p.add_argument("--only", choices=AXES, default=None,
                   help="offset the base along this one component only (arm-base frame: x forward, z down)")
    p.add_argument("--default-pose", default=None, help="npz from choose_so101_default_pose.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    gp = GraspPlanner(free_bar_rotation=not args.strict_pose, bar_slide=args.bar_slide)
    armbase0, q_default = gp.armbase_demo, gp.q_demo
    if args.default_pose:
        d = np.load(args.default_pose)
        armbase0, q_default = gp.armbase_demo @ d["base_offset"], d["q_default"]

    rng = np.random.default_rng(args.seed)
    off, ok, goals, trajs, dts, perr, rerr, slide = [], [], [], [], [], [], [], []
    for _ in range(args.num):
        xyz = rng.uniform(-args.pos_range, args.pos_range, 3)
        rpy = np.radians(rng.uniform(-1, 1, 3) * [args.tilt_range, args.tilt_range, args.yaw_range])
        if args.only:
            keep = np.arange(6) == AXES.index(args.only)
            xyz, rpy = xyz * keep[:3], rpy * keep[3:]
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = R.from_euler("xyz", rpy).as_matrix(), xyz
        armbase = armbase0 @ delta  # offset expressed in the nominal arm-base frame
        goal = gp.grasp_goal(armbase)
        res = gp.plan(armbase, q_default)
        off.append([*xyz, *np.degrees(rpy)])
        ok.append(res is not None)
        goals.append(goal)
        if res:
            trajs.append(res[0])
            dts.append(res[1])
            e = gp.errors(goal, res[0][-1])
            perr.append(e[0])
            rerr.append(e[1])
            slide.append(e[2])

    off, ok = np.array(off), np.array(ok)
    print(f"\nmode: {'full 6-DoF hand-object pose' if args.strict_pose else f'bar rotation free, slide up to {args.bar_slide * 100:.0f} cm'}, "
          f"{'chosen default pose' if args.default_pose else 'demo pose (arm stretched)'}; "
          f"base offsets +-{args.pos_range} m, yaw +-{args.yaw_range} deg, tilt +-{args.tilt_range} deg")
    print(f"planned {ok.sum()}/{len(ok)} ({100 * ok.mean():.0f}%)")
    if args.only:  # which values of the one varied component can the arm make up for?
        v = off[:, AXES.index(args.only)]
        edges = np.linspace(v.min(), v.max() + 1e-9, 7)
        scale, unit = (100, "cm") if args.only in "xyz" else (1, "deg")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            print(f"  {args.only} {lo * scale:+6.1f}..{hi * scale:+6.1f} {unit}: {ok[m].sum()}/{m.sum()}")
    else:
        dist = np.linalg.norm(off[:, :3], axis=1)
        for lo, hi in ((0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.30)):
            m = (dist >= lo) & (dist < hi)
            if m.any():
                print(f"  offset {lo * 100:.0f}-{hi * 100:.0f} cm: {ok[m].sum()}/{m.sum()}")
    if ok.any():
        dur = [(len(t) - 1) * dt for t, dt in zip(trajs, dts)]
        print(f"reached: off the target across the bar mean {np.mean(perr):.2f} mm (worst {np.max(perr):.2f}), "
              f"slide along the bar mean {np.mean(slide):.0f} mm (worst {np.max(slide):.0f}), bar misalignment "
              f"mean {np.mean(rerr):.1f} deg (worst {np.max(rerr):.1f}); duration mean {np.mean(dur):.2f} s")
    if args.out and ok.any():
        steps = max(len(t) for t in trajs)  # plain padded array: readable without pickle by any numpy
        np.savez_compressed(
            args.out, offset=off, success=ok, goal_in_base=np.array([np.array(g) for g in goals]),
            traj=np.stack([np.pad(t, ((0, steps - len(t)), (0, 0)), mode="edge") for t in trajs]).astype(np.float32),
            traj_len=np.array([len(t) for t in trajs]), dt=np.array(dts), pos_err_mm=np.array(perr),
            rot_err_deg=np.array(rerr), slide_mm=np.array(slide), q_default=np.asarray(q_default, float), handle_grasp=gp.handle_grasp,
            armbase0=armbase0, handle=gp.handle, args=np.str_(str(vars(args))))
        print("saved", args.out)
    gp.p.destroy()


if __name__ == "__main__":  # cuRobo spawns a helper process that re-imports this file
    main()
