"""Collect the arm joint commands of recorded grasp demos into one small file.

The drone RL environment (AerialWaypointTrack) replays these as the arm's motion, so the drone
learns under the motions the arm policy was trained to produce.

    python scripts/build_arm_motion_bank.py --episodes outputs/soarm101_AerialTransport_handle_grasp_demos_range_depth_fwd

Writes aetos_multiam/data/arm_motion/so101_grasp_bank.npz:
    action (E, T, 6)  joint position targets, rad (5 arm joints + jaw), sampled at `fps`
    q      (E, T, 6)  the joint positions the simulated arm reached
    length (E,)       frames of each episode (shorter ones are padded with their last frame)
    fps, joint_names
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--episodes", required=True, help="directory with ep*.npz files (searched recursively)")
p.add_argument("--out", default=str(REPO / "aetos_multiam/data/arm_motion/so101_grasp_bank.npz"))
args = p.parse_args()

files = sorted(Path(args.episodes).rglob("ep*.npz"))
assert files, f"no episodes under {args.episodes}"
act, q, fps, names = [], [], None, None
for f in files:
    e = np.load(f)
    act.append(e["action"].astype(np.float32))
    q.append(e["q"].astype(np.float32))
    fps, names = float(e["fps"]), e["joint_names"]
T = max(len(a) for a in act)


def pad(x):
    return np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)], 0)


np.savez_compressed(args.out, action=np.stack([pad(a) for a in act]), q=np.stack([pad(x) for x in q]),
                    length=np.array([len(a) for a in act]), fps=fps, joint_names=names)
print(f"{len(files)} episodes, {T} frames at {fps:g} Hz -> {args.out}")
