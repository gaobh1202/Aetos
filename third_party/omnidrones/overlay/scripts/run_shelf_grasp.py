"""Fly the trained HoverAmovDynaArm policy into the raids-am-simulator
shelf_grasp scene, hold hover at the scene's recorded drone position, then
replay a recorded button_grasp episode's arm/gripper trajectory while the
policy keeps running to reject the disturbance the arm motion creates.

Two phases in one continuous rollout:
  FLY_IN  : drone resets at target_pos + (0, +0.3, 0); policy flies it to
            target_pos (the scene's recorded ROBOT_POS); arm held at the
            dataset's frame-0 pose.
  REPLAY  : once position error and speed both stay under threshold for
            `settle_steps_required` consecutive steps (or `fly_in_timeout_steps`
            elapses), the arm's PD hold-targets are driven frame-by-frame from
            the recorded episode (30 fps, mapped onto sim control-step time),
            and the four props are teleported to their recorded pose in sync.
            The policy keeps stepping every frame throughout.

The episode data is the .npz format raids-am-simulator/raids_am_Scene/replay/
export_episode.py already produces from the LeRobot dataset (joint_rad,
gripper_m, {obj}_pos/{obj}_quat_wxyz for gearwheel/button/power_drill/tote_box).

Usage:
  OMNIDRONES_AMOV_USD_PATH=/home/benhua/OmniDrones/amov_drone_arm_1.usd \
  python scripts/run_shelf_grasp.py task=HoverShelfGrasp algo=ppo headless=false \
    checkpoint=/home/benhua/OmniDrones/wandb/run-20260601_002626-amov_drone_dynamic_arm_hover/files/checkpoint_final.pt \
    episode_npz=/path/to/episode000.npz
"""

import logging
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

# raids-am-simulator/raids_am_Scene/build_isaacsim_scene.py owns the shelf's
# ROBOT_POS/ROBOT_YAW_DEG this script treats as the fixed hover target -- same
# reuse-not-duplicate rationale as hover_shelf_grasp.py's scene construction.
_RAIDS_SCENE_DIR = Path("/home/benhua/raids-am-simulator/raids_am_Scene")
if str(_RAIDS_SCENE_DIR) not in sys.path:
    sys.path.insert(0, str(_RAIDS_SCENE_DIR))

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from setproctitle import setproctitle

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType

# build_isaacsim_scene.py does `from pxr import Gf, ...` at its own module
# top-level. pxr's real bindings are only injected into sys.path once
# SimulationApp boots (inside main(), via init_simulation_app below), so this
# import must happen AFTER that -- not here at module load time.

FILE_PATH = os.path.dirname(__file__)

OBJECT_NAMES = ("gearwheel", "button", "power_drill", "tote_box")
OBJECT_MAIN_BODY_NAME = {"button": "E_body_16", "power_drill": "power_drill"}

PHASE_FLY_IN = "FLY_IN"
PHASE_REPLAY = "REPLAY"


# ---------------------------------------------------------------------------
# Multi-mesh rigid-body resolution, copied from raids-am-simulator/
# raids_am_Scene/replay/replay_isaacsim.py's quat_to_R/quat_multiply/
# resolve_object_bodies (not imported: that script parses sys.argv and boots
# its own SimulationApp at module import time, so it can't be imported into
# an already-running Isaac Sim process).
# ---------------------------------------------------------------------------

def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _find_all_rigid_bodies(stage, root_path):
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"missing prim: {root_path}")
    return [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _resolve_object_bodies(stage, root_path, main_name_hint=None):
    from pxr import Usd, UsdGeom
    all_bodies = _find_all_rigid_bodies(stage, root_path)
    if len(all_bodies) == 1:
        return all_bodies[0], []
    main_path = f"{root_path}/{main_name_hint}" if main_name_hint else all_bodies[0]
    if main_path not in all_bodies:
        raise RuntimeError(f"{root_path}: main body hint {main_name_hint!r} not found among {all_bodies}")
    main_world = UsdGeom.Xformable(stage.GetPrimAtPath(main_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    main_pos = np.array(main_world.ExtractTranslation())
    main_quat_gf = main_world.ExtractRotationQuat()
    main_quat = np.array([main_quat_gf.GetReal(), *main_quat_gf.GetImaginary()])
    main_R_inv = _quat_to_R(main_quat).T

    siblings = []
    for path in all_bodies:
        if path == main_path:
            continue
        sib_world = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        sib_pos = np.array(sib_world.ExtractTranslation())
        sib_quat_gf = sib_world.ExtractRotationQuat()
        sib_quat = np.array([sib_quat_gf.GetReal(), *sib_quat_gf.GetImaginary()])
        delta_pos = main_R_inv @ (sib_pos - main_pos)
        delta_quat = _quat_multiply(np.array([main_quat[0], *-main_quat[1:]]), sib_quat)
        siblings.append((path, delta_pos, delta_quat))
    return main_path, siblings


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="shelf_grasp")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    checkpoint = cfg.get("checkpoint", None)
    episode_npz = cfg.get("episode_npz", None)
    if not checkpoint:
        raise ValueError("Must pass checkpoint=<path to checkpoint_final.pt>")
    if not episode_npz:
        raise ValueError("Must pass episode_npz=<path to an exported episode .npz>")

    pos_err_thresh = float(cfg.get("pos_err_thresh", 0.05))
    vel_thresh = float(cfg.get("vel_thresh", 0.1))
    settle_steps_required = int(cfg.get("settle_steps_required", 15))
    fly_in_timeout_steps = int(cfg.get("fly_in_timeout_steps", 1000))
    dataset_fps = float(cfg.get("dataset_fps", 30.0))
    post_replay_hold_steps = int(cfg.get("post_replay_hold_steps", 120))

    simulation_app = init_simulation_app(cfg)
    setproctitle("run_shelf_grasp")
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.prims import RigidPrim
    from omni_drones.utils.torch import euler_to_quaternion, quat_axis
    import build_isaacsim_scene as shelf_scene  # needs sys.path insert above + a live SimulationApp

    # ------------------------------------------------------------------
    # Load the recorded episode.
    # ------------------------------------------------------------------
    data = np.load(episode_npz)
    n_frames = len(data["frame_index"])
    logging.info("Loaded %d frames from %s", n_frames, episode_npz)

    def _arm_targets_at(i):
        return torch.tensor(
            np.concatenate([data["joint_rad"][i], [data["gripper_m"][i], -data["gripper_m"][i]]]),
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Build the env (spawns the drone + references the shelf/props scene,
    # see omni_drones/envs/single/hover_shelf_grasp.py).
    # ------------------------------------------------------------------
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    device = base_env.device

    # HoverAmovDynaArm's own max_episode_length (500 control steps) is a
    # training-time convenience limit -- nowhere near enough to fly in and
    # then replay the full recorded episode (605 frames at dataset_fps, which
    # at this sim's control_hz needs well over 1000 steps on its own). Raise
    # it so `truncated` doesn't cut the rollout short; only a real crash
    # (`terminated`) should end it early.
    _control_hz = 1.0 / (base_env.dt * base_env.substeps)
    _needed_steps = fly_in_timeout_steps + int(n_frames * _control_hz / dataset_fps) + 200
    _max_ep_override = int(cfg.get("max_episode_length_override", -1))
    base_env.max_episode_length = (
        _max_ep_override if _max_ep_override > 0
        else max(base_env.max_episode_length, _needed_steps)
    )
    logging.info(
        "max_episode_length set to %d (control_hz=%.2f, needed for full replay=%d)",
        base_env.max_episode_length, _control_hz, _needed_steps,
    )

    frame0_arm_targets = _arm_targets_at(0).to(device)
    base_env._arm_hold_targets = frame0_arm_targets

    # Fixed hover target = the shelf scene's own recorded drone position/yaw.
    target_pos = torch.tensor([list(shelf_scene.ROBOT_POS)], dtype=torch.float32, device=device)  # (1, 3)
    target_yaw_rad = math.radians(shelf_scene.ROBOT_YAW_DEG)
    init_offset = torch.tensor([0.0, 0.3, 0.0], device=device)
    fixed_init_pos = (target_pos + init_offset).unsqueeze(1)  # (1, 1, 3)
    fixed_rot = euler_to_quaternion(
        torch.tensor([[[0.0, 0.0, target_yaw_rad]]], device=device)
    )  # (1, 1, 4)
    fixed_heading = quat_axis(fixed_rot.squeeze(1), 0).unsqueeze(1)  # (1, 1, 3)

    base_env.target_pos = target_pos

    # Hover._reset_idx only ever re-orients target_vis (the green marker),
    # never repositions it -- it's spawned once in _design_scene() at a
    # hardcoded (0, 0, 2) translation that only happens to match the default
    # target_pos=(0,0,2). Since we overrode target_pos to the scene's actual
    # ROBOT_POS, the marker must be moved to match too (same technique
    # eval_hover_robustness.py's MovingTargetWrapper uses whenever target_pos
    # is changed away from that default).
    target_world_pos = (target_pos + base_env.envs_positions).unsqueeze(1)  # (num_envs, 1, 3)

    def _patched_reset_idx(self, env_ids):
        n = len(env_ids)
        self.drone._reset_idx(env_ids, self.training)
        self.drone.reset_arm_joints(env_ids, frame0_arm_targets)
        self.drone.set_world_poses(
            fixed_init_pos.expand(n, 1, 3) + self.envs_positions[env_ids].unsqueeze(1),
            fixed_rot.expand(n, 1, 4),
            env_ids,
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.target_heading[env_ids] = fixed_heading.expand(n, 1, 3)
        self.target_vis.set_world_poses(
            positions=target_world_pos[env_ids],
            orientations=fixed_rot.expand(n, 1, 4),
            env_indices=env_ids,
        )
        self.stats[env_ids] = 0.0

    base_env._reset_idx = types.MethodType(_patched_reset_idx, base_env)
    base_env.target_vis.set_world_poses(positions=target_world_pos)

    # ------------------------------------------------------------------
    # Resolve the shelf-scene props' rigid-body prims for per-frame teleport
    # (same technique as replay_isaacsim.py: resolve_object_bodies handles
    # Button/PowerDrill's extra sibling meshes).
    # ------------------------------------------------------------------
    stage = prim_utils.get_current_stage()
    objects_root = f"{base_env.template_env_ns}/Objects"
    from omni_drones.envs.single.hover_shelf_grasp import OBJECT_PRIM_NAMES

    object_prims = {}
    object_siblings = {}
    object_main_paths = {}
    for name in OBJECT_NAMES:
        root = f"{objects_root}/{OBJECT_PRIM_NAMES[name]}"
        main_path, siblings = _resolve_object_bodies(stage, root, OBJECT_MAIN_BODY_NAME.get(name))
        object_main_paths[name] = main_path
        main_prim = RigidPrim(prim_path=main_path, name=f"replay_{name}")
        main_prim.initialize()
        object_prims[name] = main_prim
        sib_prims = []
        for i, (p, dp, dq) in enumerate(siblings):
            sib_prim = RigidPrim(prim_path=p, name=f"replay_{name}_sib{i}")
            sib_prim.initialize()
            sib_prims.append((sib_prim, dp, dq))
        object_siblings[name] = sib_prims
    logging.info("Resolved prop rigid bodies: %s", object_main_paths)

    def _teleport_objects(i):
        for name in OBJECT_NAMES:
            main_pos = data[f"{name}_pos"][i]
            main_quat = data[f"{name}_quat_wxyz"][i]
            object_prims[name].set_world_pose(position=main_pos, orientation=main_quat)
            for sib_prim, delta_pos, delta_quat in object_siblings[name]:
                sib_pos = main_pos + _quat_to_R(main_quat) @ delta_pos
                sib_quat = _quat_multiply(main_quat, delta_quat)
                sib_prim.set_world_pose(position=sib_pos, orientation=sib_quat)

    _teleport_objects(0)

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
    # Policy-performance logging: arm-base joint force/torque (same mechanism
    # as eval_hover_robustness.py -- get_measured_joint_forces() returns
    # (num_envs, num_bodies, 6): [:, body_idx, :3] = force (N), [:, body_idx,
    # 3:] = torque (N*m) at the body's incoming joint, body_idx for joint1 =
    # the first arm link), plus tracking error, speed, and the policy's own
    # raw action output every step -- everything needed to judge how well the
    # policy is actually flying, not just what force the arm exerts on it.
    # ------------------------------------------------------------------
    joint_forces_out = cfg.get("joint_forces_out", None)
    _view = base_env.drone._view
    _body_names = list(_view.body_names) if hasattr(_view, "body_names") and _view.body_names else []
    _arm_body_idx = None
    for _candidate in ("joint1", "Link1", "link1", "arm_link1"):
        if _candidate in _body_names:
            _arm_body_idx = _body_names.index(_candidate)
            break
    if _arm_body_idx is None and len(_body_names) > 0:
        for _i, _bn in enumerate(_body_names):
            if not _bn.startswith("rotor") and _bn != "base_link" and _i > 0:
                _arm_body_idx = _i
                break
    logging.info(
        "Arm-base joint force logging: body_idx=%s (%s)  out=%s",
        _arm_body_idx, _body_names[_arm_body_idx] if _arm_body_idx is not None else "N/A",
        joint_forces_out,
    )
    _log_joint_forces = joint_forces_out is not None and _arm_body_idx is not None
    _force_log = []
    _force_log_saved = False

    def _save_force_log():
        if not _force_log:
            return
        steps_arr = np.array([r[0] for r in _force_log], dtype=np.int32)
        forces_arr = np.stack([r[1] for r in _force_log])
        torques_arr = np.stack([r[2] for r in _force_log])
        pos_err_arr = np.array([r[3] for r in _force_log], dtype=np.float32)
        speed_arr = np.array([r[4] for r in _force_log], dtype=np.float32)
        actions_arr = np.stack([r[5] for r in _force_log])
        np.save(joint_forces_out, {
            "steps": steps_arr,
            "forces": forces_arr,
            "torques": torques_arr,
            "pos_err": pos_err_arr,
            "speed": speed_arr,
            "actions": actions_arr,
            "body_name": _body_names[_arm_body_idx],
            "test_mode": "button_grasp",
            "replay_start_step": replay_start_step,
            "replay_finished_step": replay_finished_step,
        }, allow_pickle=True)
        logging.info(
            "Saved policy-performance log (%d steps) to %s  F_mean=[%.2f %.2f %.2f] N  "
            "pos_err_mean=%.3f  speed_mean=%.3f",
            len(_force_log), joint_forces_out,
            forces_arr[:, 0].mean(), forces_arr[:, 1].mean(), forces_arr[:, 2].mean(),
            pos_err_arr.mean(), speed_arr.mean(),
        )

    # ------------------------------------------------------------------
    # Rollout: FLY_IN -> REPLAY, held at the last frame once playback ends.
    # ------------------------------------------------------------------
    control_hz = 1.0 / (base_env.dt * base_env.substeps)
    logging.info(
        "control_hz=%.2f  dataset_fps=%.2f  target_pos=%s",
        control_hz, dataset_fps, target_pos.squeeze(0).tolist(),
    )

    phase = PHASE_FLY_IN
    settle_counter = 0
    fly_in_step = 0
    replay_start_step = None
    replay_finished_step = None
    replay_frame_idx = 0
    step = 0

    td = env.reset()
    try:
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            while simulation_app.is_running():
                td = policy(td)
                action_this_step = td[("agents", "action")][0, 0].cpu().numpy().copy()

                if phase == PHASE_REPLAY:
                    elapsed_steps = step - replay_start_step
                    frame_f = elapsed_steps * (dataset_fps / control_hz)
                    replay_frame_idx = min(int(frame_f), n_frames - 1)
                    base_env._arm_hold_targets = _arm_targets_at(replay_frame_idx).to(device)
                    _teleport_objects(replay_frame_idx)
                    if replay_finished_step is None and frame_f >= n_frames - 1:
                        replay_finished_step = step
                        logging.info(
                            "Grasp replay finished at step=%d (frame %d/%d) -- holding %d more "
                            "steps then stopping.",
                            step, replay_frame_idx, n_frames - 1, post_replay_hold_steps,
                        )

                td = env.step(td)

                pos_err = torch.norm(base_env.rpos[0, 0]).item()
                speed = torch.norm(base_env.drone.vel[0, 0, :3]).item()

                if _log_joint_forces and not _force_log_saved:
                    jf = _view.get_measured_joint_forces()
                    if jf is not None:
                        _force_log.append((
                            step,
                            jf[0, _arm_body_idx, :3].cpu().numpy().copy(),
                            jf[0, _arm_body_idx, 3:].cpu().numpy().copy(),
                            pos_err,
                            speed,
                            action_this_step,
                        ))

                terminated = td["next"]["terminated"]
                if terminated.any():
                    _z = base_env.drone.pos[0, 0, 2].item()
                    _up = base_env.drone.up[0, 0, 2].item()
                    logging.info(
                        "Episode terminated (crash) at step=%d -- z=%.3f up_z=%.3f "
                        "pos_err=%.3f phase=%s replay_frame=%d/%d. Stopping rollout.",
                        step, _z, _up, pos_err, phase, replay_frame_idx, n_frames - 1,
                    )
                    if _log_joint_forces and not _force_log_saved:
                        _save_force_log()
                        _force_log_saved = True
                    break

                if replay_finished_step is not None and (step - replay_finished_step) >= post_replay_hold_steps:
                    logging.info(
                        "Grasp replay completed and held for %d steps -- stopping rollout "
                        "(pos_err=%.3f, speed=%.3f).",
                        post_replay_hold_steps, pos_err, speed,
                    )
                    if _log_joint_forces and not _force_log_saved:
                        _save_force_log()
                        _force_log_saved = True
                    break

                if phase == PHASE_FLY_IN:
                    fly_in_step += 1
                    if pos_err < pos_err_thresh and speed < vel_thresh:
                        settle_counter += 1
                    else:
                        settle_counter = 0
                    if settle_counter >= settle_steps_required or fly_in_step >= fly_in_timeout_steps:
                        phase = PHASE_REPLAY
                        replay_start_step = step
                        logging.info(
                            "FLY_IN -> REPLAY at step=%d  pos_err=%.3f  by_timeout=%s",
                            step, pos_err, fly_in_step >= fly_in_timeout_steps,
                        )
                elif step % 100 == 0:
                    logging.info(
                        "REPLAY frame=%d/%d  pos_err=%.3f  speed=%.3f",
                        replay_frame_idx, n_frames - 1, pos_err, speed,
                    )

                td = td["next"]
                step += 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user at step=%d.", step)
    finally:
        if _log_joint_forces and not _force_log_saved:
            _save_force_log()
        simulation_app.close()


if __name__ == "__main__":
    main()
