"""Pick the arm-base pose and bent default posture the grasp should start from.

The demos reach the handle with the arm stretched out (all joints zero), which
leaves no margin outward. This scans base positions in the arm's vertical plane
(x forward, z down, relative to the demo's base), keeps those from which the handle
can be grasped, and takes the one farthest from any failure: the base may then be
off by that margin in any in-plane direction. The default posture is that grasp
pulled back along the fingers, so the arm still has a reach to perform.

    conda activate armplan
    python scripts/arm_planning/choose_so101_default_pose.py
"""
import argparse
import os.path as osp

import numpy as np

from so101_common import ROOT, T_DRONE_ARMBASE, T
from so101_planner import GraspPlanner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--retreat", type=float, default=0.08, help="how far the default posture holds the gripper back, m")
    p.add_argument("--out", default=osp.join(ROOT, "aetos_multiam/data/aerial_transport/so101_default_pose.npz"))
    args = p.parse_args()

    # The grasp is one fixed object-centric pose (position AND orientation, as in the demos):
    # no free rotation about the bar, no sliding along it.
    gp = GraspPlanner(free_bar_rotation=False)
    xs = np.arange(-0.15, 0.30 + 1e-9, args.step)
    zs = np.arange(-0.25, 0.20 + 1e-9, args.step)
    ok = np.zeros((len(xs), len(zs)), bool)
    q_end = {}
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            res = gp.plan(gp.armbase_demo @ T((x, 0, z), (1, 0, 0, 0)), gp.q_demo)
            if res:
                ok[i, j], q_end[i, j] = True, res[0][-1]
    print("graspable base positions (rows: x, + = towards the handle; columns: z from %+.0f to %+.0f cm, "
          "+ = base lower)" % (zs[0] * 100, zs[-1] * 100))
    for i in range(len(xs)):
        print("  x %+5.1f  " % (xs[i] * 100) + "".join("#" if o else "." for o in ok[i]))

    # Margin of a graspable cell = distance to the nearest failed (or unscanned) cell.
    bad = np.argwhere(~np.pad(ok, 1))
    margin = np.zeros(ok.shape)
    for i, j in np.argwhere(ok):
        margin[i, j] = np.min(np.hypot(*(bad - [i + 1, j + 1]).T)) * args.step
    i, j = np.unravel_index(np.argmax(margin), margin.shape)
    base_offset = T((xs[i], 0, zs[j]), (1, 0, 0, 0))
    armbase = gp.armbase_demo @ base_offset
    q_grasp = q_end[i, j]
    print(f"chosen base: x {xs[i] * 100:+.1f} cm, z {zs[j] * 100:+.1f} cm from the demo's; "
          f"margin {margin[i, j] * 100:.1f} cm in the vertical plane (demo's own base: {margin[np.argmin(abs(xs)), np.argmin(abs(zs))] * 100:.1f} cm)")

    # Default posture: the same grasp held back along the fingers (grasp-frame +z points into the palm).
    goal = gp.grasp_goal(armbase) @ T((0, 0, args.retreat), (1, 0, 0, 0))
    res = gp.plan(armbase, q_grasp, goal)
    assert res is not None, "no retreat posture; lower --retreat"
    q_default = res[0][-1]
    g = gp.grasp_goal(armbase)[:3, 3]
    print("grasp posture   (deg):", np.round(np.degrees(q_grasp), 1))
    print("default posture (deg):", np.round(np.degrees(q_default), 1))
    print(f"handle from the arm base: {g[0] * 100:.1f} cm forward, {g[2] * 100:.1f} cm below "
          f"(demo: {gp.grasp_goal(gp.armbase_demo)[0, 3] * 100:.1f}, {gp.grasp_goal(gp.armbase_demo)[2, 3] * 100:.1f})")
    np.savez(args.out, base_offset=base_offset, q_default=q_default, q_grasp=q_grasp, margin_m=margin[i, j],
             grid_x=xs, grid_z=zs, graspable=ok,
             # What a simulation needs, all in the handle frame (origin at the bar centre, box axes,
             # turned 180 deg about z for the second robot): drone base_link pose, grasp-frame pose.
             drone_in_handle=np.linalg.inv(gp.handle) @ armbase @ np.linalg.inv(T_DRONE_ARMBASE),
             grasp_in_handle=gp.handle_grasp, retreat_m=args.retreat)
    print("saved", args.out)
    gp.p.destroy()


if __name__ == "__main__":  # cuRobo spawns a helper process that re-imports this file
    main()
