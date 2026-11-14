"""Serve a trained LeRobot policy to a simulator running in another Python environment.

    conda activate lerobot
    python scripts/serve_lerobot_policy.py outputs/train/act_so101_handle_grasp/checkpoints/last/pretrained_model

The client (scripts/collect_arm_grasp_demo.py --policy localhost:6000) sends "reset" at the start
of an episode, then one observation per control step: {"state": [floats], "shape": (h, w),
"<camera>": raw RGB bytes, ...}; the reply is the action as a list of floats. Only builtin types
cross the socket, so the two environments' numpy / torch versions do not have to match.
"""
import argparse
from multiprocessing.connection import Listener

import numpy as np
import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action

p = argparse.ArgumentParser()
p.add_argument("checkpoint", help=".../checkpoints/<step>/pretrained_model")
p.add_argument("--port", type=int, default=6000)
p.add_argument("--task", default="Grasp the handle of the box and lift it.")
args = p.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
device = torch.device(cfg.device)
with Listener(("localhost", args.port)) as listener:
    print(f"[serve] {cfg.type} from {args.checkpoint} on localhost:{args.port}", flush=True)
    while True:
        with listener.accept() as conn:
            try:
                while True:
                    msg = conn.recv()
                    if msg == "reset":
                        policy.reset()
                        continue
                    h, w = msg.pop("shape")
                    obs = {"observation.state": np.asarray(msg.pop("state"), dtype=np.float32)}
                    for cam, raw in msg.items():
                        obs[f"observation.images.{cam}"] = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
                    a = predict_action(obs, policy, device, pre, post, use_amp=False, task=args.task)
                    conn.send(a.reshape(-1).tolist())
            except EOFError:
                print("[serve] client left", flush=True)
