"""Replay the cuRobo grasp plans in Isaac Sim and measure where the gripper ends up.

Each environment pins drone 0 at one planned arm-base pose next to the carry box
(pose and velocity rewritten every physics step, so only the arm moves), drives the
arm along the planned joint trajectory, holds, and compares the grasp point with the
handle bar. Box-arm contacts are filtered in this task, so this checks that plan,
frames and simulated arm agree (and with --drive, PD tracking under gravity), not
collisions or the jaw closing.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python scripts/arm_planning/exec_so101_grasp_isaac.py [--slowdown 3] [--num-envs 16]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
p = argparse.ArgumentParser()
p.add_argument("--plans", default=str(REPO / "aetos_multiam/data/aerial_transport/arm_grasp_plans.npz"))
p.add_argument("--num-envs", type=int, default=16)
p.add_argument("--slowdown", type=float, default=3.0, help="play the plan this many times slower")
p.add_argument("--hold", type=float, default=0.6, help="seconds to settle after the plan ends")
p.add_argument("--drive", action="store_true",
               help="command the arm through its PD drives. Off by default (joint positions are written "
                    "directly, a purely kinematic replay) because shoulder_pan's drive is dead in this scene "
                    "(docs/ARM_DISTURBANCE.md) and writing only that joint's position disturbs the others")
args = p.parse_args()

os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from aetos_multiam import init_simulation_app  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=str(REPO / "cfg")):
    cfg = compose(config_name="train", overrides=["task=AerialTransportTrack", "headless=true",
                                                  f"++task.env.num_envs={args.num_envs}"])
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
app = init_simulation_app(cfg)
try:
    import torch
    from scipy.spatial.transform import Rotation as R

    from aetos_multiam.environments import make as make_env
    from so101_common import GRASP_LOCAL, T_DRONE_ARMBASE, T

    z = np.load(args.plans)
    dts = z["dt"]
    trajs = [t[:n] for t, n in zip(z["traj"], z["traj_len"])]
    if "drone_in_handle" in z:  # feasible_base_set.py: drone poses already in the handle frame
        bases = z["drone_in_handle"][z["feasible"]]
        handle_grasp, plan_err = z["grasp_in_handle"], np.zeros(len(trajs))
    else:  # plan_so101_grasp.py: offsets from a nominal arm base, in the demo's world frame
        bases = []
        for o in z["offset"][z["success"]]:
            delta = np.eye(4)
            delta[:3, :3], delta[:3, 3] = R.from_euler("xyz", np.radians(o[3:])).as_matrix(), o[:3]
            bases.append(np.linalg.inv(z["handle"]) @ z["armbase0"] @ delta @ np.linalg.inv(T_DRONE_ARMBASE))
        handle_grasp, plan_err = z["handle_grasp"], z["pos_err_mm"]
    env = make_env("AerialTransportTrack", cfg, headless=True)
    env.reset()
    N, dev, dt = env.num_envs, env.device, float(env.dt)
    am = env.aerial_manipulators[0]
    drones = [a.drone for a in env.aerial_manipulators]
    view = drones[0]._view
    names = list(view._physics_view.shared_metatype.link_names)
    gi = names.index("gripper")
    arm_idx = torch.as_tensor(am.teleop_joint_indices, device=dev)

    def mat(pos, quat_wxyz):
        return T(pos.cpu().numpy(), quat_wxyz.cpu().numpy())

    bp, bq = env.box_view.get_world_poses()
    handle = [mat(bp[e], bq[e]) @ T((-0.42, 0.0, 0.34), (1, 0, 0, 0)) for e in range(N)]
    other = [d.get_world_poses() for d in drones[1:]]
    zero6 = torch.zeros(N, 1, 6, device=dev)
    rows, per_joint = [], []
    for c0 in range(0, len(trajs), N):
        ids = [min(c0 + e, len(trajs) - 1) for e in range(N)]  # last chunk repeats its final plan
        pos, quat = np.zeros((N, 3)), np.zeros((N, 4))
        for e, i in enumerate(ids):
            m = handle[e] @ bases[i]
            pos[e], quat[e] = m[:3, 3], np.roll(R.from_matrix(m[:3, :3]).as_quat(), 1)
        pos = torch.as_tensor(pos, device=dev, dtype=torch.float32).unsqueeze(1)
        quat = torch.as_tensor(quat, device=dev, dtype=torch.float32).unsqueeze(1)
        dur = [(len(trajs[i]) - 1) * dts[i] * args.slowdown for i in ids]
        q0 = torch.zeros(N, 1, 6, device=dev)
        q0[..., :5] = torch.as_tensor(z["q_default"], device=dev, dtype=torch.float32)
        q0[..., 5] = 1.5  # jaw open
        view.set_joint_positions(q0, joint_indices=arm_idx)
        view.set_joint_velocities(torch.zeros_like(q0), joint_indices=arm_idx)
        lag = np.zeros(N)
        for k in range(int((max(dur) + args.hold) / dt)):
            tgt = q0.clone()
            for e, i in enumerate(ids):
                s = min(k * dt / dur[e], 1.0) * (len(trajs[i]) - 1)
                a, w = min(int(s), len(trajs[i]) - 2), s - min(int(s), len(trajs[i]) - 2)
                tgt[e, 0, :5] = torch.as_tensor((1 - w) * trajs[i][a] + w * trajs[i][a + 1], device=dev)
            view.set_joint_position_targets(tgt, joint_indices=arm_idx)
            if not args.drive:
                view.set_joint_positions(tgt, joint_indices=arm_idx)
            drones[0].set_world_poses(pos, quat)
            drones[0].set_velocities(zero6)
            for d, (op, oq) in zip(drones[1:], other):
                d.set_world_poses(op, oq)
                d.set_velocities(zero6)
            env.sim.step(False)
            q = view.get_joint_positions()[:, 0][:, arm_idx]
            lag = np.maximum(lag, (q[:, :5] - tgt[:, 0, :5]).abs().max(-1).values.cpu().numpy())
        lt = view._physics_view.get_link_transforms().reshape(N, len(names), 7)[:, gi].cpu().numpy()
        qf = (q[:, :5] - tgt[:, 0, :5]).abs().max(-1).values.cpu().numpy()
        per_joint.append((q[:len(trajs) - c0, :5] - tgt[:len(trajs) - c0, 0, :5]).cpu().numpy())
        for e, i in enumerate(ids[:len(trajs) - c0]):
            g = T(lt[e, :3], np.roll(lt[e, 3:], 1)) @ T(GRASP_LOCAL, (1, 0, 0, 0))
            rel = np.linalg.inv(handle[e]) @ g  # grasp frame in the handle frame (bar along y)
            want = handle_grasp
            bar_want, bar_got = want[:3, :3].T @ [0, 1, 0], rel[:3, :3].T @ [0, 1, 0]
            rows.append([np.hypot(rel[0, 3], rel[2, 3]) * 1000, abs(rel[1, 3]) * 1000,
                         np.degrees(np.arccos(np.clip(bar_want @ bar_got, -1, 1))),
                         np.degrees(qf[e]), np.degrees(lag[e]), plan_err[i]])
        print(f"[exec] {min(c0 + N, len(trajs))}/{len(trajs)} plans replayed", flush=True)
    r = np.array(rows)
    for name, col, unit in (("grasp point to bar axis", 0, "mm"), ("slide along the bar", 1, "mm"),
                            ("gripper-bar misalignment", 2, "deg"), ("final joint error", 3, "deg"),
                            ("largest joint lag while moving", 4, "deg"), ("planner's own position error", 5, "mm")):
        print(f"[exec] {name:32s} mean {r[:, col].mean():6.2f}  worst {r[:, col].max():6.2f} {unit}")
    pj = np.degrees(np.abs(np.concatenate(per_joint)))
    print("[exec] final error per joint (pan, lift, elbow, wrist_flex, wrist_roll), deg: mean",
          np.round(pj.mean(0), 2), "worst", np.round(pj.max(0), 2))
    ok = (r[:, 0] < 10) & (r[:, 1] < 90)  # bar (r = 15 mm) inside the 68 mm jaw gap, clear of the posts at +-110 mm
    print(f"[exec] bar ends up inside the open jaw: {ok.sum()}/{len(r)}  ({'PD drive' if args.drive else 'kinematic replay'}, slowdown x{args.slowdown})")
    print("BUILD done")
except Exception:
    import traceback
    traceback.print_exc()
finally:
    app.close()
