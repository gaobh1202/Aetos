"""Turn the trained HoverAmovDynaArm hover policy into a circular-trajectory
tracker, with NO retraining: every control step the policy's fixed-point
target (target_pos / target_heading) is overridden to the next point on a
circle, so the same "fly to a point and hold" policy is asked to continuously
chase a moving point instead.

Two phases in one continuous rollout:
  SETTLE : target held fixed at the circle's start point (angle=0); the
           drone flies there and settles (position error and speed both
           under threshold for `settle_steps_required` consecutive steps, or
           `settle_timeout_steps` elapses as a fallback).
  TRACK  : the target advances around the circle at a constant angular rate
           (`circle_period_sec` seconds per lap); target_heading is set to
           the circle's tangent direction at that point, so the drone is
           asked to face the direction of travel. Runs for `num_laps` laps,
           then holds at the final point for `post_track_hold_steps` before
           the script exits on its own.

The arm is held at a neutral fixed pose throughout (not driven by the
trained sinusoid/replay), so this test isolates position/heading tracking
from arm-disturbance rejection -- the two things this checkpoint was
originally evaluated on separately.

Usage:
  OMNIDRONES_AMOV_USD_PATH=/home/benhua/OmniDrones/amov_drone_arm_1.usd \
  python scripts/run_circle_track.py headless=false \
    checkpoint=/path/to/checkpoint_final.pt \
    circle_radius=2.0 circle_period_sec=10.0 num_laps=3 \
    out=/path/to/circle_track.npy
"""

import logging
import math
import os
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from setproctitle import setproctitle

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType

FILE_PATH = os.path.dirname(__file__)

PHASE_SETTLE = "SETTLE"
PHASE_TRACK = "TRACK"


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="circle_track")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    checkpoint = cfg.get("checkpoint", None)
    if not checkpoint:
        raise ValueError("Must pass checkpoint=<path to checkpoint_final.pt>")

    circle_center = torch.tensor(list(cfg.get("circle_center", [0.0, 0.0, 2.0])), dtype=torch.float32)
    circle_radius = float(cfg.get("circle_radius", 2.0))
    circle_period_sec = float(cfg.get("circle_period_sec", 10.0))
    circle_direction = 1.0 if float(cfg.get("circle_direction", 1)) >= 0 else -1.0
    num_laps = float(cfg.get("num_laps", 3))

    pos_err_thresh = float(cfg.get("pos_err_thresh", 0.05))
    vel_thresh = float(cfg.get("vel_thresh", 0.1))
    settle_steps_required = int(cfg.get("settle_steps_required", 15))
    settle_timeout_steps = int(cfg.get("settle_timeout_steps", 1000))
    post_track_hold_steps = int(cfg.get("post_track_hold_steps", 150))

    simulation_app = init_simulation_app(cfg)
    setproctitle("run_circle_track")
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.utils.torch import euler_to_quaternion

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    device = base_env.device
    circle_center = circle_center.to(device)

    control_hz = 1.0 / (base_env.dt * base_env.substeps)
    period_steps = max(1, round(circle_period_sec * control_hz))
    angular_rate = circle_direction * (2.0 * math.pi / period_steps)  # rad / control step
    linear_speed = 2.0 * math.pi * circle_radius / circle_period_sec
    logging.info(
        "control_hz=%.2f  period_steps=%d (%.1fs/lap)  linear_speed=%.2f m/s  "
        "radius=%.2f  center=%s  num_laps=%.1f",
        control_hz, period_steps, circle_period_sec, linear_speed,
        circle_radius, circle_center.tolist(), num_laps,
    )

    needed_track_steps = int(math.ceil(num_laps * period_steps))
    needed_steps = settle_timeout_steps + needed_track_steps + post_track_hold_steps + 200
    max_ep_override = int(cfg.get("max_episode_length_override", -1))
    base_env.max_episode_length = (
        max_ep_override if max_ep_override > 0
        else max(base_env.max_episode_length, needed_steps)
    )
    logging.info("max_episode_length set to %d (needed=%d)", base_env.max_episode_length, needed_steps)

    # Hold the arm at a neutral fixed pose -- this test isolates
    # position/heading tracking from arm-disturbance rejection.
    base_env._arm_hold_enabled = True
    base_env._arm_trajectory_enabled = False
    base_env._arm_motion_enabled = False
    base_env._arm_replay_enabled = False
    base_env._arm_hold_targets = torch.zeros(base_env.drone.num_arm_joints, device=device)
    base_env._arm_hold_zero_velocity_each_step = False

    def _circle_point(angle: float) -> torch.Tensor:
        offset = torch.tensor(
            [math.cos(angle), math.sin(angle), 0.0], device=device
        ) * circle_radius
        return circle_center + offset  # (3,)

    def _circle_tangent(angle: float) -> torch.Tensor:
        d = torch.tensor(
            [-math.sin(angle), math.cos(angle), 0.0], device=device
        ) * circle_direction
        return torch.nn.functional.normalize(d, dim=-1, eps=1e-6)  # (3,)

    def _set_target(angle: float):
        pos = _circle_point(angle)  # (3,)
        tangent = _circle_tangent(angle)  # (3,)
        base_env.target_pos = pos.unsqueeze(0)  # (1, 3), broadcasts over envs
        base_env.target_heading[:] = tangent.view(1, 1, 3)
        yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        rot = euler_to_quaternion(torch.tensor([[[0.0, 0.0, yaw]]], device=device))  # (1,1,4)
        world_pos = (pos.view(1, 1, 3) + base_env.envs_positions.unsqueeze(1))  # (num_envs,1,3)
        base_env.target_vis.set_world_poses(
            positions=world_pos, orientations=rot.expand(base_env.num_envs, 1, 4)
        )
        return pos, tangent

    # Spawn the drone at the circle's start point (angle=0) instead of the
    # task's default random-offset init distribution, same monkeypatch
    # pattern as run_shelf_grasp.py's _patched_reset_idx.
    start_pos = _circle_point(0.0)
    start_tangent = _circle_tangent(0.0)
    start_yaw = math.atan2(float(start_tangent[1]), float(start_tangent[0]))
    start_rot = euler_to_quaternion(torch.tensor([[[0.0, 0.0, start_yaw]]], device=device))

    def _patched_reset_idx(self, env_ids):
        n = len(env_ids)
        self.drone._reset_idx(env_ids, self.training)
        self.drone.reset_arm_joints(env_ids, base_env._arm_hold_targets)
        self.drone.set_world_poses(
            start_pos.view(1, 1, 3).expand(n, 1, 3) + self.envs_positions[env_ids].unsqueeze(1),
            start_rot.expand(n, 1, 4),
            env_ids,
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.target_heading[env_ids] = start_tangent.view(1, 1, 3).expand(n, 1, 3)
        self.stats[env_ids] = 0.0

    base_env._reset_idx = types.MethodType(_patched_reset_idx, base_env)

    # ------------------------------------------------------------------
    # Policy.
    # ------------------------------------------------------------------
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()
    env.set_seed(0)

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=device,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt, strict=False)
    logging.info("Loaded checkpoint: %s", checkpoint)

    # ------------------------------------------------------------------
    # Logging.
    # ------------------------------------------------------------------
    out_path = cfg.get("out", None)
    _log = []
    _log_saved = False

    def _save_log():
        if not _log:
            return
        steps_arr = np.array([r[0] for r in _log], dtype=np.int32)
        drone_pos_arr = np.stack([r[1] for r in _log])
        target_pos_arr = np.stack([r[2] for r in _log])
        pos_err_arr = np.array([r[3] for r in _log], dtype=np.float32)
        speed_arr = np.array([r[4] for r in _log], dtype=np.float32)
        heading_cos_arr = np.array([r[5] for r in _log], dtype=np.float32)
        actions_arr = np.stack([r[6] for r in _log])
        angle_arr = np.array([r[7] for r in _log], dtype=np.float32)
        np.save(out_path, {
            "steps": steps_arr,
            "drone_pos": drone_pos_arr,
            "target_pos": target_pos_arr,
            "pos_err": pos_err_arr,
            "speed": speed_arr,
            "heading_cos": heading_cos_arr,
            "actions": actions_arr,
            "angle": angle_arr,
            "circle_center": circle_center.cpu().numpy(),
            "circle_radius": circle_radius,
            "circle_period_sec": circle_period_sec,
            "track_start_step": track_start_step,
            "track_finished_step": track_finished_step,
        }, allow_pickle=True)
        logging.info(
            "Saved circle-tracking log (%d steps) to %s  pos_err_mean=%.3f  speed_mean=%.3f",
            len(_log), out_path, pos_err_arr.mean(), speed_arr.mean(),
        )

    # ------------------------------------------------------------------
    # Rollout: SETTLE -> TRACK, held at the final point once num_laps done.
    # ------------------------------------------------------------------
    phase = PHASE_SETTLE
    settle_counter = 0
    settle_step = 0
    angle = 0.0
    track_start_step = None
    track_finished_step = None
    step = 0

    td = env.reset()
    _set_target(angle)
    try:
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            while simulation_app.is_running():
                td = policy(td)
                action_this_step = td[("agents", "action")][0, 0].cpu().numpy().copy()

                td = env.step(td)

                drone_pos = base_env.drone.pos[0, 0].cpu().numpy().copy()
                target_pos_now = base_env.target_pos.reshape(-1, 3)[0].cpu().numpy().copy()
                pos_err = torch.norm(base_env.rpos[0, 0]).item()
                speed = torch.norm(base_env.drone.vel[0, 0, :3]).item()
                heading_cos = torch.sum(
                    base_env.drone.heading[0, 0] * base_env.target_heading[0, 0]
                ).item()

                if out_path is not None and not _log_saved:
                    _log.append((step, drone_pos, target_pos_now, pos_err, speed,
                                  heading_cos, action_this_step, angle))

                terminated = td["next"]["terminated"]
                if terminated.any():
                    logging.info(
                        "Episode terminated (crash) at step=%d phase=%s pos_err=%.3f. Stopping.",
                        step, phase, pos_err,
                    )
                    if out_path is not None and not _log_saved:
                        _save_log()
                        _log_saved = True
                    break

                if track_finished_step is not None and (step - track_finished_step) >= post_track_hold_steps:
                    logging.info(
                        "Completed %.1f laps and held for %d steps -- stopping (pos_err=%.3f, speed=%.3f).",
                        num_laps, post_track_hold_steps, pos_err, speed,
                    )
                    if out_path is not None and not _log_saved:
                        _save_log()
                        _log_saved = True
                    break

                if phase == PHASE_SETTLE:
                    settle_step += 1
                    if pos_err < pos_err_thresh and speed < vel_thresh:
                        settle_counter += 1
                    else:
                        settle_counter = 0
                    if settle_counter >= settle_steps_required or settle_step >= settle_timeout_steps:
                        phase = PHASE_TRACK
                        track_start_step = step
                        logging.info(
                            "SETTLE -> TRACK at step=%d  pos_err=%.3f  by_timeout=%s",
                            step, pos_err, settle_step >= settle_timeout_steps,
                        )
                else:
                    angle += angular_rate
                    laps_done = abs(angle) / (2.0 * math.pi)
                    if track_finished_step is None and laps_done >= num_laps:
                        track_finished_step = step
                        logging.info(
                            "Completed %.1f laps at step=%d -- holding %d more steps.",
                            num_laps, step, post_track_hold_steps,
                        )
                    if step % 100 == 0:
                        logging.info(
                            "TRACK laps=%.2f/%.1f  pos_err=%.3f  speed=%.3f  heading_cos=%.3f",
                            laps_done, num_laps, pos_err, speed, heading_cos,
                        )

                _set_target(angle)
                td = td["next"]
                step += 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user at step=%d.", step)
    finally:
        if out_path is not None and not _log_saved:
            _save_log()
        simulation_app.close()


if __name__ == "__main__":
    main()
