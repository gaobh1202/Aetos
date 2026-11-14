"""Robustness evaluation for HoverAmovDynaArm policy.

Two test modes (controlled by --test):
  target          : target position changes every `target_change_interval` steps
                    while arm follows the trained fixed trajectory.
  arm_perturb     : target is fixed; arm trajectory parameters are randomized
                    per-episode to test generalization beyond the training trajectory.
  arm_random      : target is fixed; arm trajectory is fully re-sampled each episode
                    from independent uniform ranges per joint (amplitude, frequency,
                    phase all randomized within configurable bounds).
  target_arm      : moving target + arm_perturb combined.
  target_arm_random : moving target + fully random arm trajectory combined.
                    Tests policy robustness when the arm deviates arbitrarily from
                    the trained trajectory while the target also moves.

Usage:
  OMNIDRONES_AMOV_USD_PATH=/home/benhua/OmniDrones/amov_drone_arm_1.usd \
  python scripts/eval_hover_robustness.py \
    task=HoverAmovDynaArm algo=ppo headless=false \
    checkpoint=/path/to/checkpoint_final.pt \
    test=arm_random

  # Tune the random ranges:
  test=arm_random \
    arm_rand_amp_max="[1.2,0.8,0.7,0.5,0.5,0.3]" \
    arm_rand_freq_min=0.1 arm_rand_freq_max=0.8 \
    arm_rand_phase_random=true
"""

import logging
import math
import os
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault(
    "OMNI_DRONES_HYDRA_SEARCHPATH",
    str(_REPO_ROOT / "cfg"),
)

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from setproctitle import setproctitle

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from omni_drones.utils.torchrl import EpisodeStats
from omni_drones.utils.torchrl.transforms import ravel_composite
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType


FILE_PATH = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Wrapper: moving target
# ---------------------------------------------------------------------------

class MovingTargetWrapper:
    """Patches a Hover env so the target position changes periodically."""

    def __init__(self, base_env, change_interval: int, pos_range: float = 1.5):
        self.env = base_env
        self.change_interval = change_interval
        self.pos_range = pos_range
        self._step_count = 0

    def _randomize_target(self):
        env = self.env
        num_envs = env.num_envs
        device = env.device
        # Random XY offset, fixed Z = 2.0
        xy = (torch.rand(num_envs, 2, device=device) * 2 - 1) * self.pos_range
        z = torch.full((num_envs, 1), 2.0, device=device)
        new_pos = torch.cat([xy, z], dim=-1)  # (num_envs, 3)
        # target_pos is (1, 3) broadcast; set per-env via target_vis world poses
        # We store per-env target in a (num_envs, 1, 3) tensor
        env.target_pos = new_pos.unsqueeze(1)  # (num_envs, 1, 3)
        # Move the visual marker
        world_pos = new_pos + env.envs_positions  # (num_envs, 3)
        env.target_vis.set_world_poses(positions=world_pos.unsqueeze(1))
        logging.info(
            "[MovingTarget] step=%d  new target pos (env0): %.2f %.2f %.2f",
            self._step_count,
            float(new_pos[0, 0]), float(new_pos[0, 1]), float(new_pos[0, 2]),
        )

    def step_hook(self):
        self._step_count += 1
        if self._step_count % self.change_interval == 0:
            self._randomize_target()


# ---------------------------------------------------------------------------
# Wrapper: arm trajectory perturbation
# ---------------------------------------------------------------------------

class ArmPerturbWrapper:
    """Randomizes arm trajectory amplitudes/frequencies at each episode reset."""

    def __init__(self, base_env, perturb_scale: float = 0.5):
        """
        perturb_scale: fraction of original amplitude/frequency to perturb by.
          0.0 = no perturbation (same as training)
          1.0 = amplitude/frequency can vary ±100% of original value
        """
        self.env = base_env
        self.scale = perturb_scale
        # Store original values
        self._orig_amp = base_env._arm_traj_amplitudes.clone()
        self._orig_freq = base_env._arm_traj_frequencies.clone()

    def reset_hook(self, env_ids: torch.Tensor):
        env = self.env
        n = env.drone.num_arm_joints
        device = env.device

        # Sample per-env perturbation multipliers in [1-scale, 1+scale]
        amp_mult = 1.0 + (torch.rand(n, device=device) * 2 - 1) * self.scale
        freq_mult = 1.0 + (torch.rand(n, device=device) * 2 - 1) * self.scale

        new_amp = (self._orig_amp * amp_mult).clamp(min=0.0)
        new_freq = (self._orig_freq * freq_mult).clamp(min=0.05)

        env._arm_traj_amplitudes = new_amp
        env._arm_traj_frequencies = new_freq

        logging.info(
            "[ArmPerturb] new amplitudes=%s  frequencies=%s",
            [f"{v:.3f}" for v in new_amp.tolist()],
            [f"{v:.3f}" for v in new_freq.tolist()],
        )


# ---------------------------------------------------------------------------
# Wrapper: fully random arm trajectory (independent per-joint sampling)
# ---------------------------------------------------------------------------

class ArmRandomTrajWrapper:
    """Re-samples a completely new sinusoidal trajectory at each episode reset.

    Unlike ArmPerturbWrapper (which perturbs around the training trajectory),
    this wrapper draws amplitude, frequency, and phase independently from
    uniform ranges for each joint, producing visually distinct motions.

    Per-joint amplitude limits (rad) default to conservative fractions of the
    physical joint range.  Gripper joints (endleft/endright) are always kept
    at zero.

    Config keys (all optional, passed via cfg):
      arm_rand_amp_max   : list[float] or float — per-joint max amplitude (rad).
                           If scalar, applied to all active joints.
                           Default: [1.2, 0.8, 0.7, 0.5, 0.5, 0.3] for joint1-6.
      arm_rand_amp_min   : float — minimum amplitude (rad), default 0.1.
      arm_rand_freq_min  : float — minimum frequency (Hz), default 0.1.
      arm_rand_freq_max  : float — maximum frequency (Hz), default 0.8.
      arm_rand_phase_random : bool — randomize phase in [0, 2π], default True.
    """

    # Physical joint limits for airbot_play (conservative, in rad)
    _JOINT_AMP_MAX_DEFAULT = [1.2, 0.8, 0.7, 0.5, 0.5, 0.3]

    def __init__(self, base_env, cfg):
        self.env = base_env
        n_total = base_env.drone.num_arm_joints   # includes gripper DOFs
        n_active = min(6, n_total)                # joint1–joint6 only
        device = base_env.device

        # Parse amp_max — per-joint list or scalar
        amp_max_cfg = cfg.get("arm_rand_amp_max", None)
        if amp_max_cfg is None:
            amp_max_list = self._JOINT_AMP_MAX_DEFAULT[:n_active]
        elif isinstance(amp_max_cfg, (int, float)):
            amp_max_list = [float(amp_max_cfg)] * n_active
        else:
            amp_max_list = [float(v) for v in amp_max_cfg][:n_active]

        self._amp_max   = torch.tensor(amp_max_list, device=device)
        self._amp_min   = float(cfg.get("arm_rand_amp_min",  0.1))
        self._freq_min  = float(cfg.get("arm_rand_freq_min", 0.1))
        self._freq_max  = float(cfg.get("arm_rand_freq_max", 0.8))
        self._rand_phase = bool(cfg.get("arm_rand_phase_random", True))
        self._n_active  = n_active
        self._n_total   = n_total

    def reset_hook(self, env_ids: torch.Tensor):
        env = self.env
        device = env.device
        n = self._n_active

        # Sample amplitude: uniform in [amp_min, amp_max] per joint
        amp_range = (self._amp_max - self._amp_min).clamp(min=0.0)
        new_amp = self._amp_min + torch.rand(n, device=device) * amp_range

        # Sample frequency: uniform in [freq_min, freq_max]
        new_freq = self._freq_min + torch.rand(n, device=device) * (
            self._freq_max - self._freq_min
        )

        # Sample phase: uniform in [0, 2π] or keep zero
        if self._rand_phase:
            new_phase = torch.rand(n, device=device) * (2.0 * math.pi)
        else:
            new_phase = torch.zeros(n, device=device)

        # Pad gripper joints with zeros
        if self._n_total > n:
            pad = torch.zeros(self._n_total - n, device=device)
            new_amp   = torch.cat([new_amp,   pad])
            new_freq  = torch.cat([new_freq,  pad])
            new_phase = torch.cat([new_phase, pad])

        env._arm_traj_amplitudes  = new_amp
        env._arm_traj_frequencies = new_freq
        env._arm_traj_phases      = new_phase

        logging.info(
            "[ArmRandom] amp=%s  freq=%s  phase=%s",
            [f"{v:.3f}" for v in new_amp[:self._n_active].tolist()],
            [f"{v:.3f}" for v in new_freq[:self._n_active].tolist()],
            [f"{v:.2f}" for v in new_phase[:self._n_active].tolist()],
        )


# ---------------------------------------------------------------------------
# Patched env subclasses
# ---------------------------------------------------------------------------

def make_moving_target_env(base_env, change_interval: int, pos_range: float):
    wrapper = MovingTargetWrapper(base_env, change_interval, pos_range)

    orig_compute = base_env._compute_state_and_obs.__func__

    def patched_compute(self):
        wrapper.step_hook()
        return orig_compute(self)

    import types
    base_env._compute_state_and_obs = types.MethodType(patched_compute, base_env)
    return base_env, wrapper


def make_arm_perturb_env(base_env, perturb_scale: float):
    wrapper = ArmPerturbWrapper(base_env, perturb_scale)

    orig_reset = base_env._reset_idx.__func__

    def patched_reset(self, env_ids):
        orig_reset(self, env_ids)
        wrapper.reset_hook(env_ids)

    import types
    base_env._reset_idx = types.MethodType(patched_reset, base_env)
    return base_env, wrapper


def make_arm_random_env(base_env, cfg):
    wrapper = ArmRandomTrajWrapper(base_env, cfg)

    orig_reset = base_env._reset_idx.__func__

    def patched_reset(self, env_ids):
        orig_reset(self, env_ids)
        wrapper.reset_hook(env_ids)

    import types
    base_env._reset_idx = types.MethodType(patched_reset, base_env)
    # Trigger once immediately so the first episode already uses a random traj.
    all_ids = torch.arange(base_env.num_envs, device=base_env.device)
    wrapper.reset_hook(all_ids)
    return base_env, wrapper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=FILE_PATH, config_name="eval")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    checkpoint = cfg.get("checkpoint", None)
    test_mode = cfg.get("test", "target")
    target_change_interval = int(cfg.get("target_change_interval", 100))
    target_pos_range = float(cfg.get("target_pos_range", 1.5))
    arm_perturb_scale = float(cfg.get("arm_perturb_scale", 0.5))
    num_episodes = int(cfg.get("num_episodes", 20))
    max_episode_length_override = int(cfg.get("max_episode_length", -1))
    # force_log_steps: how many global steps to record joint forces.
    # -1 = auto: for target/target_arm modes, cover num_target_changes full
    # target-change cycles; for other modes, record for one full episode (500).
    _force_log_steps_cfg = int(cfg.get("force_log_steps", -1))
    num_target_changes = int(cfg.get("num_target_changes", 5))

    simulation_app = init_simulation_app(cfg)
    setproctitle(f"eval_{test_mode}")
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # Override max_episode_length if requested.
    if max_episode_length_override > 0:
        base_env.max_episode_length = max_episode_length_override
        logging.info("max_episode_length overridden to %d", max_episode_length_override)

    # Disable early termination during eval so episodes always run to
    # max_episode_length.  We wrap _compute_reward_and_done to zero out
    # `terminated` and rewrite `done = truncated` in the returned TensorDict.
    # This lets the drone recover from transients (e.g. after a target change)
    # instead of resetting mid-episode and corrupting the force log.
    disable_termination = bool(cfg.get("disable_termination", True))
    if disable_termination:
        import types
        _orig_reward_done = base_env._compute_reward_and_done.__func__

        def _patched_reward_done(self):
            td = _orig_reward_done(self)
            truncated = td.get("truncated")
            td.set("terminated", torch.zeros_like(td.get("terminated")))
            td.set("done", truncated.clone())
            return td

        base_env._compute_reward_and_done = types.MethodType(
            _patched_reward_done, base_env
        )
        logging.info(
            "Early termination DISABLED — episodes run to max_episode_length=%d.",
            base_env.max_episode_length,
        )

    # Verify arm state matches training config and warn if not.
    _arm_hold = getattr(base_env, "_arm_hold_enabled", False)
    _arm_traj = getattr(base_env, "_arm_trajectory_enabled", False)
    _arm_motion = getattr(base_env, "_arm_motion_enabled", False)
    logging.info(
        "Arm state: hold=%s  trajectory=%s  motion=%s",
        _arm_hold, _arm_traj, _arm_motion,
    )

    # Apply test wrapper
    if test_mode == "target":
        logging.info(
            "Test mode: MOVING TARGET  change_interval=%d  pos_range=%.1f",
            target_change_interval, target_pos_range,
        )
        base_env, _ = make_moving_target_env(
            base_env, target_change_interval, target_pos_range
        )
    elif test_mode == "arm_perturb":
        if not _arm_traj:
            raise RuntimeError(
                "arm_trajectory is not enabled in this task config. "
                "Use task=HoverAmovDynaArm for arm_perturb mode."
            )
        logging.info(
            "Test mode: ARM PERTURBATION  perturb_scale=%.2f", arm_perturb_scale
        )
        base_env, _ = make_arm_perturb_env(base_env, arm_perturb_scale)
    elif test_mode == "target_arm":
        if not _arm_traj:
            raise RuntimeError(
                "arm_trajectory is not enabled in this task config. "
                "Use task=HoverAmovDynaArm for target_arm mode."
            )
        logging.info(
            "Test mode: MOVING TARGET + ARM PERTURBATION  change_interval=%d  pos_range=%.1f  perturb_scale=%.2f",
            target_change_interval, target_pos_range, arm_perturb_scale,
        )
        base_env, _ = make_moving_target_env(
            base_env, target_change_interval, target_pos_range
        )
        base_env, _ = make_arm_perturb_env(base_env, arm_perturb_scale)
    elif test_mode == "target_arm_random":
        if not _arm_traj:
            raise RuntimeError(
                "arm_trajectory is not enabled in this task config. "
                "Use task=HoverAmovDynaArm for target_arm_random mode."
            )
        logging.info(
            "Test mode: MOVING TARGET + RANDOM ARM TRAJECTORY  change_interval=%d  pos_range=%.1f  freq=[%.2f, %.2f] Hz",
            target_change_interval, target_pos_range,
            float(cfg.get("arm_rand_freq_min", 0.1)),
            float(cfg.get("arm_rand_freq_max", 0.8)),
        )
        base_env, _ = make_moving_target_env(
            base_env, target_change_interval, target_pos_range
        )
        base_env, _ = make_arm_random_env(base_env, cfg)
    elif test_mode == "arm_random":
        if not _arm_traj:
            raise RuntimeError(
                "arm_trajectory is not enabled in this task config. "
                "Use task=HoverAmovDynaArm for arm_random mode."
            )
        logging.info(
            "Test mode: ARM RANDOM TRAJECTORY  freq=[%.2f, %.2f] Hz  phase_random=%s",
            float(cfg.get("arm_rand_freq_min", 0.1)),
            float(cfg.get("arm_rand_freq_max", 0.8)),
            str(cfg.get("arm_rand_phase_random", True)),
        )
        base_env, _ = make_arm_random_env(base_env, cfg)
    elif test_mode == "fixed":
        logging.info("Test mode: FIXED (arm and target unchanged from training config)")
    else:
        raise ValueError(
            f"Unknown test mode: {test_mode!r}. "
            "Use 'fixed', 'target', 'arm_perturb', 'arm_random', 'target_arm', or 'target_arm_random'."
        )

    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(0)

    # Build policy
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )

    # Load checkpoint
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=base_env.device)
        policy.load_state_dict(ckpt, strict=False)
        logging.info("Loaded checkpoint: %s", checkpoint)
    else:
        logging.warning("No checkpoint provided — running with random policy.")

    # ------------------------------------------------------------------
    # Arm-base joint force logging setup
    # ------------------------------------------------------------------
    # get_measured_joint_forces returns (num_envs, num_bodies, 6):
    #   [:, body_idx, :3] = force (N),  [:, body_idx, 3:] = torque (N·m)
    # body_idx for joint1 = joint_index_in_dof_list + 1  (index 0 = base link)
    # We want the incoming joint force at joint1 (first arm joint = body index 5
    # for AmovDrone: 4 rotor bodies + base_link + joint1...).
    # Resolve at runtime from body_names.
    _view = base_env.drone._view
    _body_names = list(_view.body_names) if hasattr(_view, "body_names") and _view.body_names else []
    logging.info("ArticulationView body_names: %s", _body_names)

    # Find body index for "joint1" link (the first arm link).
    # Isaac Sim body_names typically match link names in the USD.
    _arm_body_idx = None
    for _candidate in ("joint1", "Link1", "link1", "arm_link1"):
        if _candidate in _body_names:
            _arm_body_idx = _body_names.index(_candidate)
            break
    if _arm_body_idx is None and len(_body_names) > 0:
        # Fallback: first non-rotor, non-base body
        for _i, _bn in enumerate(_body_names):
            if not _bn.startswith("rotor") and _bn != "base_link" and _i > 0:
                _arm_body_idx = _i
                break
    logging.info(
        "Arm-base joint force: using body_idx=%s (%s)",
        _arm_body_idx,
        _body_names[_arm_body_idx] if _arm_body_idx is not None else "N/A",
    )

    _log_joint_forces = _arm_body_idx is not None
    _force_log = []   # list of (step, force_xyz, torque_xyz) for env 0

    # Determine how many global steps to record joint forces.
    if _force_log_steps_cfg > 0:
        _force_log_steps = _force_log_steps_cfg
    elif test_mode in ("target", "target_arm", "target_arm_random"):
        # Cover num_target_changes full target-change cycles so the log
        # captures both the transient (drone re-adjusting) and steady-state
        # phases for each new target position.
        _force_log_steps = target_change_interval * num_target_changes
    else:
        # fixed / arm_perturb: one full episode
        _force_log_steps = 500
    logging.info("Joint force log will record %d global steps.", _force_log_steps)

    # ------------------------------------------------------------------
    # Collect episodes — manual step loop to capture per-step joint forces
    # ------------------------------------------------------------------
    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collected_episodes = 0
    _global_step = 0
    _print_interval = 50   # print to console every N steps

    logging.info("Starting evaluation rollouts (%d episodes)...", num_episodes)

    td = env.reset()
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        while collected_episodes < num_episodes:
            # Policy forward pass
            td = policy(td)
            td = env.step(td)

            # --- Capture joint force at arm-base junction ---
            if _log_joint_forces and _global_step < _force_log_steps:
                jf = _view.get_measured_joint_forces()  # (num_envs, num_bodies, 6)
                if jf is not None:
                    f_xyz = jf[0, _arm_body_idx, :3]   # force  (N),   env 0
                    t_xyz = jf[0, _arm_body_idx, 3:]   # torque (N·m), env 0
                    _force_log.append((
                        _global_step,
                        f_xyz.cpu().numpy().copy(),
                        t_xyz.cpu().numpy().copy(),
                    ))
                    if _global_step % _print_interval == 0:
                        logging.info(
                            "[JointForce] step=%4d  F=[%6.2f %6.2f %6.2f] N  "
                            "T=[%6.3f %6.3f %6.3f] N·m",
                            _global_step,
                            float(f_xyz[0]), float(f_xyz[1]), float(f_xyz[2]),
                            float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2]),
                        )

            _global_step += 1

            # Collect stats on episode end
            next_td = td.get("next")
            done = next_td.get("done")
            if done.any():
                episode_stats.add(td.to_tensordict())
                collected_episodes += int(done.sum().item())
                logging.info("Collected %d / %d episodes", collected_episodes, num_episodes)

            td = next_td

    # Summarize
    if len(episode_stats) > 0:
        stats = episode_stats.pop()
        print("\n" + "=" * 60)
        print(f"Evaluation results  [mode={test_mode}]")
        print("=" * 60)
        for k, v in stats.items(True, True):
            key = ".".join(k) if isinstance(k, tuple) else k
            print(f"  {key:40s}: {torch.mean(v.float()).item():.4f}")
        print("=" * 60)

    # Save joint force log to npy
    if _log_joint_forces and len(_force_log) > 0:
        import numpy as np
        steps   = np.array([r[0] for r in _force_log], dtype=np.int32)
        forces  = np.stack([r[1] for r in _force_log])   # (T, 3)
        torques = np.stack([r[2] for r in _force_log])   # (T, 3)
        out_path = f"joint_forces_{test_mode}.npy"
        np.save(out_path, {
            "steps":   steps,
            "forces":  forces,   # Fx Fy Fz  (N)
            "torques": torques,  # Tx Ty Tz  (N·m)
            "body_name": _body_names[_arm_body_idx],
            "test_mode": test_mode,
            "target_change_interval": target_change_interval if test_mode in ("target", "target_arm") else None,
        }, allow_pickle=True)
        logging.info(
            "Saved joint force log (%d steps) to %s  "
            "F_mean=[%.2f %.2f %.2f] N  T_mean=[%.3f %.3f %.3f] N·m",
            len(_force_log), out_path,
            forces[:, 0].mean(), forces[:, 1].mean(), forces[:, 2].mean(),
            torques[:, 0].mean(), torques[:, 1].mean(), torques[:, 2].mean(),
        )
        print(f"\nJoint force data saved to: {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
