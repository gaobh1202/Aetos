"""Serve a trained LeRobot policy to a simulator running in another Python environment.

    conda activate lerobot
    python scripts/serve_lerobot_policy.py outputs/train/act_so101_handle_grasp/checkpoints/last/pretrained_model

The client (scripts/collect_arm_grasp_demo.py --policy localhost:6000) sends "reset" at the start
of an episode, then one observation per control step: {"state": [floats] (the joints, plus the
drone's own state if the policy was trained with it -- the client's --drone-state), "shape": (h, w),
"<camera>": raw RGB bytes, ...}; the reply is the action as a list of floats. Only builtin types
cross the socket, so the two environments' numpy / torch versions do not have to match.
"""
import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

p = argparse.ArgumentParser()
p.add_argument("checkpoint", help=".../checkpoints/<step>/pretrained_model")
p.add_argument("--port", type=int, default=6000)
p.add_argument("--task", default="Grasp the handle of the box and lift it.")
p.add_argument("--n-action-steps", type=int, default=None,
               help="actions executed from each predicted chunk before predicting again (default: as trained)")
p.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
               help="ACT: predict every step and average the overlapping chunks, weights exp(-COEFF * age); 0.01 in the paper")
p.add_argument("--num-inference-steps", type=int, default=None, metavar="N",
               help="diffusion: denoise in N DDIM steps instead of the full training schedule (the same weights "
                    "serve both samplers, so this also works on a checkpoint trained with DDPM)")
args = p.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
if args.temporal_ensemble is not None:
    cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.temporal_ensemble, 1
elif args.n_action_steps:
    cfg.n_action_steps = args.n_action_steps
if args.num_inference_steps:
    if getattr(cfg, "objective", None) == "flow_matching":
        cfg.num_integration_steps = args.num_inference_steps
    else:
        cfg.noise_scheduler_type, cfg.num_inference_steps = "DDIM", args.num_inference_steps
policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
device = torch.device(cfg.device)
latency = []
policy.to(device)


def predict_action(obs):  # numpy observation (state vector, HxWx3 uint8 images) -> numpy action
    batch = {"task": args.task}
    for k, v in obs.items():
        t = torch.from_numpy(v)
        if "image" in k:
            t = t.permute(2, 0, 1).float() / 255.0
        batch[k] = t.unsqueeze(0).to(device)
    with torch.inference_mode():
        return post(policy.select_action(pre(batch))).squeeze(0).cpu().numpy()


with Listener(("localhost", args.port)) as listener:
    print(f"[serve] {cfg.type} (n_action_steps {cfg.n_action_steps}, temporal ensemble "
          f"{getattr(cfg, 'temporal_ensemble_coeff', None)}) from {args.checkpoint} on localhost:{args.port}", flush=True)
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
                    t0 = time.perf_counter()
                    a = predict_action(obs)
                    latency.append(time.perf_counter() - t0)
                    conn.send(a.reshape(-1).tolist())
            except EOFError:
                lat = np.sort(latency)  # most calls only pop a queued action; the slow ones run the network
                print(f"[serve] client left; {len(lat)} calls, inference (slowest {len(lat) // max(cfg.n_action_steps, 1)}): "
                      f"{1000 * lat[len(lat) - max(len(lat) // max(cfg.n_action_steps, 1), 1):].mean():.0f} ms each", flush=True)
                latency.clear()
