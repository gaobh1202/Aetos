"""Turn the episodes of collect_arm_grasp_demo.py into a LeRobot dataset.

    conda activate lerobot
    python scripts/export_arm_grasp_lerobot.py --episodes outputs/arm_grasp_demos --out outputs/lerobot/so101_handle_grasp

State = the six joint angles (five arm joints + jaw), action = the expert's clean joint targets,
two camera streams stored as video.

Then train ACT on it (about 1 h 45 min for 40k steps on a 16 GB GPU; pyav because torchcodec may not load):

    lerobot-train --dataset.repo_id=aetos/so101_handle_grasp --dataset.root=outputs/lerobot/so101_handle_grasp \\
        --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.push_to_hub=false \\
        --policy.chunk_size=50 --policy.n_action_steps=25 --steps=40000 --batch_size=16 --num_workers=6 \\
        --save_freq=10000 --eval_freq=0 --wandb.enable=false --output_dir=outputs/train/act_so101_handle_grasp

and test it in the simulator: scripts/serve_lerobot_policy.py + scripts/collect_arm_grasp_demo.py --policy.
"""
import argparse
import glob
import os.path as osp

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--episodes", required=True, help="directory searched recursively for ep*.npz")
p.add_argument("--out", required=True, help="dataset root (must not exist)")
p.add_argument("--repo-id", default="aetos/so101_handle_grasp")
p.add_argument("--task", default="Grasp the handle of the box and lift it.")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

files = sorted(glob.glob(osp.join(args.episodes, "**", "ep*.npz"), recursive=True))[:args.limit]
assert files, f"no episodes under {args.episodes}"
first = np.load(files[0])
joints = [str(j) for j in first["joint_names"]]
cams = sorted(k[len("image_"):] for k in first.files if k.startswith("image_"))
h, w = first[f"image_{cams[0]}"].shape[1:3]
features = {
    "observation.state": {"dtype": "float32", "shape": (len(joints),), "names": joints},
    "action": {"dtype": "float32", "shape": (len(joints),), "names": joints},
    **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
       for c in cams},
}
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=int(round(float(first["fps"]))), features=features,
                           root=args.out, robot_type="so101_follower", use_videos=True)
frames = 0
for f in files:
    z = np.load(f)
    assert bool(z["success"]), f
    d = {k: z[k] for k in ("q", "action", *(f"image_{c}" for c in cams))}  # decompress once, not per frame
    for t in range(len(d["q"])):
        ds.add_frame({"observation.state": d["q"][t], "action": d["action"][t], "task": args.task,
                      **{f"observation.images.{c}": d[f"image_{c}"][t] for c in cams}})
    ds.save_episode()
    frames += len(d["q"])
ds.finalize()
print(f"exported {len(files)} episodes, {frames} frames, cameras {cams} -> {args.out}")
