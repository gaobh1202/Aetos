"""AerialTransportTrack -- learn the DRONE controller of AerialTransport (CTBR).

Curriculum stage 0 of the whole-body-controller plan (DexMachina-style: start
with the hard couplings switched off, turn them on later):

* both robots fly from ONE shared policy (identical airframes); per agent the
  action is PX4's offboard interface -- body rates + collective thrust,
  ``[p, q, r] / (220, 220, 200) deg/s`` and ``2 * thrust_frac - 1`` --
  closed by ``PX4RateController`` every physics substep (250 Hz);
* the reference is a recorded demonstration
  (``scripts/collect_aerial_transport_demo.py``): every episode starts at a
  random time of a random demo (reference-state initialisation), the drones
  are placed in the demo's state there, and the reward is for tracking the
  demo's drone trajectory from that point on;
* the arm is NOT learned: it replays the demo's joint targets, with its
  dynamics frozen (``arm_dynamics_residual``) -- curriculum knob 1;
* the box replays the demo's pose (no gravity, no arm contacts); its weight
  reaches the drones as ``payload_scale`` x half the box weight on each
  ``base_link`` while the demo had it lifted -- curriculum knob 2. Unlike the
  demo's expert, the policy gets no mass feedforward: compensating it is part
  of what it has to learn.

Not modelled yet (later stages): the arm's full dynamics, a physical box and
the internal force between the two robots through it.
"""

import glob
import math
import os
from typing import List

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.utils.torch import (
    euler_to_quaternion, quat_axis, quat_mul, quat_rotate, quat_rotate_inverse, quaternion_to_euler, quaternion_to_rotation_matrix,
)
from omni_drones.utils.torchrl import AgentSpec

from aetos_multiam.controllers.drone.px4_rate_controller import denormalize_ctbr
from aetos_multiam.objects import CarryBox
from aetos_multiam.scenes import Arena, WarehouseArena
from aetos_multiam.utils import freeze_arm_dynamics

from .aerial_transport import TABLE_TOP_Z_M, AerialTransport

#: Future reference offsets (control steps) shown to the policy.
LOOKAHEAD = (0, 8, 16, 32)
G = 9.81
#: Handle-bar centre in the gripper link frame (same as scripts/collect_aerial_transport_demo.py).
GRASP_LOCAL = (0.0108, -0.0004, -0.0682)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw / 2
    return torch.stack([half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()], -1)


class AerialTransportTrack(AerialTransport):
    def __init__(self, cfg, headless):
        t = cfg.task
        if t.get("drone_action", "rotor") != "ctbr" or not t.get("filter_box_arm_contacts", False):
            raise ValueError("AerialTransportTrack needs task.drone_action=ctbr and "
                             "task.filter_box_arm_contacts=true (see cfg/task/AerialTransportTrack.yaml).")
        self.payload_scale = float(t.get("payload_scale", 1.0))
        self.arm_residual = float(t.get("arm_dynamics_residual", 0.1))
        self.payload_at = str(t.get("payload_at", "base"))
        assert self.payload_at in ("base", "grasp"), self.payload_at
        self.reset_pos_noise = float(t.get("reset_pos_noise_m", 0.02))
        self.reset_rot_noise = math.radians(float(t.get("reset_rot_noise_deg", 2.0)))
        self.term_pos_err = float(t.get("terminate_pos_error_m", 0.5))
        self.term_tilt = math.radians(float(t.get("terminate_tilt_deg", 60.0)))
        self.rw = dict(t.get("reward", {}))
        self.demo_skip = int(t.get("demo_skip_steps", 63))
        self._load_demos(t.demo_dir)
        # UJITSO's cooked-mesh service deadlocks PhysX start-up at >= 8 envs (gripper convex decomposition).
        import carb
        carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
        super().__init__(cfg, headless)

    # -- demos -------------------------------------------------------------

    def _build_arena(self):
        if self.cfg.task.get("warehouse", False):
            return super()._build_arena()
        # Training scene: shared ground plane only (tables/box come from _load_objects).
        arena = Arena()
        arena.spawn_zone_centers = WarehouseArena.spawn_zone_centers
        return arena

    def _load_model(self):
        paths = super()._load_model()
        if not self.cfg.task.get("visuals", False):
            # Headless training: invisible prims are not loaded by the RTX renderer (RAM, build time).
            from pxr import UsdGeom
            import omni.isaac.core.utils.prims as prim_utils
            UsdGeom.Imageable(prim_utils.get_prim_at_path(self.template_env_ns)).MakeInvisible()
        return paths

    def _load_demos(self, demo_dir: str) -> None:
        demo_dir = demo_dir if os.path.isabs(demo_dir) else os.path.join(
            os.path.dirname(__file__), "..", "..", "..", demo_dir)
        files = sorted(glob.glob(os.path.join(demo_dir, "ep*.npz")))
        demos = [np.load(f) for f in files]
        demos = [d for d in demos if bool(d["success"]) and str(d["drone_action_mode"]) == "ctbr"]
        if not demos:
            raise FileNotFoundError(f"No successful CTBR demos in {demo_dir}")
        lens = [len(d["t"]) - self.demo_skip for d in demos]
        T = max(lens)

        def pad(x):  # drop the start transient, repeat the last frame up to T
            x = x[self.demo_skip:]
            return np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)], 0)

        def stack(key):
            return torch.as_tensor(np.stack([pad(d[key]) for d in demos]), dtype=torch.float32)

        def agents(key):  # (D, T, 2, ...)
            return torch.stack([stack(f"a{i}_{key}") for i in range(2)], 2)

        box_z = stack("box_pos")[..., 2]
        attached = stack("attached")
        self.demo = {
            "len": torch.tensor(lens),
            "pos": agents("drone_pos"), "quat": agents("drone_quat"), "vel": agents("drone_vel"),
            "yaw": agents("setpoint_yaw"), "action": agents("action")[..., :4],
            "thrust": agents("thrust_frac"), "arm_target": agents("arm_target"), "arm_q": agents("arm_q"),
            "box_pos": stack("box_pos"), "box_quat": stack("box_quat"),
            # The demo expert's own setpoints (for a closed-loop expert check).
            "sp_pos": agents("setpoint_pos"), "sp_vel": agents("setpoint_vel"),
            # Fraction of the box's weight the robots carry: none while it rests on a table.
            "load": attached * ((box_z - TABLE_TOP_Z_M) / 0.01).clamp(0.0, 1.0),
        }
        self.demo_dt = float(demos[0]["dt"])
        self.num_demos = len(demos)

    # -- setup -------------------------------------------------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        self.demo = {k: v.to(self.device) for k, v in self.demo.items()}
        if abs(self.demo_dt - self.dt * self.substeps) > 1e-6:
            raise ValueError(f"demo control dt {self.demo_dt} != env control dt {self.dt * self.substeps}")
        self.box_view = self.carry_box._view
        self.box_view.disable_gravities()
        self._load = torch.zeros(self.num_envs, self.num_agents, 3, device=self.device)
        for i, am in enumerate(self.aerial_manipulators):
            freeze_arm_dynamics(am, residual=self.arm_residual)
            # The SO101 jaw's authored drive (kp 0.35) cannot hold its frozen link.
            jaw = am.teleop_joint_indices[5:6]
            am.drone._view.set_gains(kps=torch.full((self.num_envs, 1), 10.0, device=self.device),
                                     kds=torch.full((self.num_envs, 1), 0.5, device=self.device),
                                     joint_indices=jaw)
            # shoulder_pan's authored drive (kp 2.8 N m/rad, tuned for a table-mounted arm)
            # lets the whole arm swing about the yaw axis on a flying base.
            pan_gains = self.cfg.task.get("shoulder_pan_gains", None)
            if pan_gains:
                am.drone._view.set_gains(kps=torch.full((self.num_envs, 1), float(pan_gains[0]), device=self.device),
                                         kds=torch.full((self.num_envs, 1), float(pan_gains[1]), device=self.device),
                                         joint_indices=am.teleop_joint_indices[0:1])
            orig = am.drone.base_link.apply_forces_and_torques_at_pos
            view = am.drone._view
            L, g = view.num_bodies, list(view.body_names).index("gripper")
            com_g = view.get_body_coms()[0].reshape(-1, L, 3)[:, g].to(self.device)
            grasp_local = torch.tensor([GRASP_LOCAL], device=self.device).expand(self.num_envs, 3)
            rows = torch.arange(self.num_envs, device=self.device) * L

            # Payload enters through the one force write apply_action makes: on base_link's CoM
            # ("base"), or on the gripper link at the handle, so the arm carries it ("grasp").
            def patched(forces=None, torques=None, positions=None, is_global=True, _i=i, _orig=orig,
                        _view=view, _L=L, _g=g, _com=com_g):
                load = self._load[:, _i].reshape(-1, 3)
                if self.payload_at == "base" or forces is None:
                    return _orig(None if forces is None else forces + load, torques, positions, is_global=is_global)
                lt = _view._physics_view.get_link_transforms().reshape(-1, _L, 7)[:, _g]
                q = torch.cat([lt[:, 6:7], lt[:, 3:6]], -1)  # PhysX xyzw -> wxyz
                arm = quat_rotate(q, grasp_local) - quat_rotate(q, _com)  # link CoM -> handle, world
                f = torch.zeros(self.num_envs * _L, 3, device=self.device)
                tq = torch.zeros_like(f)
                f[rows], tq[rows] = forces, torques  # root link is body 0
                f[rows + _g], tq[rows + _g] = load, torch.cross(arm, load, dim=-1)
                _view._physics_view.apply_forces_and_torques_at_position(
                    force_data=f, torque_data=tq, position_data=None,
                    indices=torch.arange(self.num_envs, device=self.device), is_global=True)

            am.drone.base_link.apply_forces_and_torques_at_pos = patched
        self.demo_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.t0 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(self.num_envs, self.num_agents, 4, device=self.device)
        self.prev_action = torch.zeros_like(self.last_action)

    #: Per-episode statistics a subclass adds to the common ones.
    EXTRA_STATS = ()

    def _obs_dim(self) -> int:
        return 9 + 3 + 3 + 3 * len(LOOKAHEAD) + 3 + 2 + 4 + 1 + 6

    def _setup_observables(self) -> None:
        self.obs_dim = self._obs_dim()
        n = self.num_agents
        self.observation_spec = CompositeSpec({
            "agents": {"observation": UnboundedContinuousTensorSpec((n, self.obs_dim))},
            "stats": {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "pos_error": UnboundedContinuousTensorSpec(1),
                "yaw_error_deg": UnboundedContinuousTensorSpec(1),
                "ref_progress": UnboundedContinuousTensorSpec(1),
                "terminated_early": UnboundedContinuousTensorSpec(1),
                **{k: UnboundedContinuousTensorSpec(1) for k in self.EXTRA_STATS},
            },
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {"action": BoundedTensorSpec(-1, 1, (n, 4))},
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {"reward": UnboundedContinuousTensorSpec((n, 1))},
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        self.stats = self.observation_spec["stats"].zero()

    # -- reference lookup ----------------------------------------------------

    def _ref_index(self, offset: int = 0, env_ids=None) -> torch.Tensor:
        ids = slice(None) if env_ids is None else env_ids
        t = self.t0[ids] + self.progress_buf[ids].long() + offset
        return torch.minimum(t, self.demo["len"][self.demo_idx[ids]] - 1)

    def _ref(self, key: str, offset: int = 0, env_ids=None) -> torch.Tensor:
        ids = slice(None) if env_ids is None else env_ids
        return self.demo[key][self.demo_idx[ids], self._ref_index(offset, env_ids)]

    # -- reset ---------------------------------------------------------------

    def _reset(self, tensordict, **kwargs):
        if not getattr(self, "_warmed_up", False):
            # The first state write after scene creation is only partly applied by PhysX (the drones
            # get a ~3 rad/s kick on the first step); one throwaway step, then reset for real.
            self._warmed_up = True
            super()._reset(None)
            self.sim.step(False)
        return super()._reset(tensordict, **kwargs)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.demo_idx[env_ids] = torch.randint(0, self.num_demos, (n,), device=self.device)
        horizon = int(self.cfg.task.get("min_episode_horizon", 100))
        max_t0 = (self.demo["len"][self.demo_idx[env_ids]] - horizon).clamp_min(1)
        self.t0[env_ids] = (torch.rand(n, device=self.device) * max_t0).long()
        self.progress_buf[env_ids] = 0
        origin = self.envs_positions[env_ids]
        for i, am in enumerate(self.aerial_manipulators):
            d = am.drone
            d._reset_idx(env_ids, train=True)
            pos = self._ref("pos", env_ids=env_ids)[:, i] + self.reset_pos_noise * torch.randn(n, 3, device=self.device)
            noise_rpy = self.reset_rot_noise * torch.randn(n, 3, device=self.device)
            rot = quat_mul(self._ref("quat", env_ids=env_ids)[:, i], euler_to_quaternion(noise_rpy))
            d.set_world_poses((pos + origin).unsqueeze(1), rot.unsqueeze(1), env_indices=env_ids)
            d.set_velocities(self._ref("vel", env_ids=env_ids)[:, i].unsqueeze(1), env_indices=env_ids)
            ndof = d._view.num_dof
            d._view.set_joint_velocities(torch.zeros(n, 1, ndof, device=self.device), env_indices=env_ids,
                                         joint_indices=torch.arange(ndof, device=self.device))
            # Start the motors where the demo had them (thrust is linear in throttle^2).
            frac = self._ref("thrust", env_ids=env_ids)[:, i].reshape(n, 1, 1)
            d.throttle[env_ids] = frac.sqrt().expand(-1, 1, d.num_rotors)
            idx = am.teleop_joint_indices
            q = self._ref("arm_q", env_ids=env_ids)[:, i].unsqueeze(1)
            d._view.set_joint_positions(q, env_indices=env_ids, joint_indices=idx)
            d._view.set_joint_position_targets(self._ref("arm_target", env_ids=env_ids)[:, i].unsqueeze(1),
                                               env_indices=env_ids, joint_indices=idx)
            self.rate_controllers[i].reset(env_ids)
        self.box_view.set_world_poses(self._ref("box_pos", env_ids=env_ids) + origin,
                                      self._ref("box_quat", env_ids=env_ids), env_indices=env_ids)
        self.box_view.set_velocities(torch.zeros(n, 6, device=self.device), env_indices=env_ids)
        a0 = self._ref("action", env_ids=env_ids)
        self.last_action[env_ids] = a0
        self.prev_action[env_ids] = a0
        self.stats[env_ids] = 0.0

    # -- step --------------------------------------------------------------

    def _pre_sim_step(self, tensordict: TensorDict) -> None:
        action = tensordict[("agents", "action")].clamp(-1, 1)
        if not getattr(self, "_substep_started", False):
            # Once per control step: what the policy did, and what the demo's
            # arm and box do now.
            self._substep_started = True
            self.prev_action = self.last_action.clone()
            self.last_action = action.clone()
            load = self._ref("load") * self.payload_scale * CarryBox.mass_kg * G / self.num_agents
            self._load[..., 2] = -load.unsqueeze(-1)
            for i, am in enumerate(self.aerial_manipulators):
                am.drone._view.set_joint_position_targets(self._ref("arm_target")[:, i].unsqueeze(1),
                                                          joint_indices=am.teleop_joint_indices)
            # Where the demo's box is at the END of this step.
            self.box_view.set_world_poses(self._ref("box_pos", 1) + self.envs_positions, self._ref("box_quat", 1))
            v = torch.zeros(self.num_envs, 6, device=self.device)
            v[:, :3] = (self._ref("box_pos", 1) - self._ref("box_pos")) / (self.dt * self.substeps)
            self.box_view.set_velocities(v)
        for i, am in enumerate(self.aerial_manipulators):
            rate_sp, thrust = denormalize_ctbr(action[:, i])
            state = am.drone.get_state()[:, 0, :13]
            am.apply_action(self.rate_controllers[i](state, rate_sp, thrust).unsqueeze(1))

    def _post_sim_step(self, tensordict: TensorDict) -> None:
        self._substep_started = False

    # -- observation / reward ------------------------------------------------

    def _compute_state_and_obs(self) -> TensorDict:
        states = torch.stack([am.drone.get_state()[:, 0, :13] for am in self.aerial_manipulators], 1)
        pos, rot, vel, angvel = states[..., :3], states[..., 3:7], states[..., 7:10], states[..., 10:13]
        yaw = quaternion_to_euler(rot)[..., 2]
        h = _yaw_quat(yaw)  # heading frame: same policy for robots facing +x and -x

        def to_h(v):
            return quat_rotate_inverse(h.reshape(-1, 4), v.reshape(-1, 3)).reshape(v.shape)

        R = quaternion_to_rotation_matrix(quat_mul(_yaw_quat(-yaw).reshape(-1, 4), rot.reshape(-1, 4)))
        body_rate = quat_rotate_inverse(rot.reshape(-1, 4), angvel.reshape(-1, 3)).reshape(angvel.shape)
        rel = [to_h(self._ref("pos", k) - pos) for k in LOOKAHEAD]
        ref_vel = to_h(self._ref("vel")[..., :3])
        yaw_err = self._ref("yaw") - yaw
        arm_q = torch.stack([am.drone._view.get_joint_positions()[:, 0][:, am.teleop_joint_indices]
                             for am in self.aerial_manipulators], 1)
        load = self._ref("load").unsqueeze(-1).expand(-1, self.num_agents).unsqueeze(-1)
        obs = torch.cat([R.reshape(*rot.shape[:-1], 9), body_rate, to_h(vel), *rel, ref_vel,
                         yaw_err.sin().unsqueeze(-1), yaw_err.cos().unsqueeze(-1),
                         self.last_action, load, arm_q], -1)

        self._pos_err = rel[0].norm(dim=-1)  # (N, 2)
        self._vel_err = (vel - self._ref("vel")[..., :3]).norm(dim=-1)
        self._yaw_err = torch.atan2(yaw_err.sin(), yaw_err.cos()).abs()
        up_ref = quat_axis(self._ref("quat").reshape(-1, 4), 2).reshape(*rot.shape[:-1], 3)
        up = quat_axis(rot.reshape(-1, 4), 2).reshape(*rot.shape[:-1], 3)
        self._tilt_err = torch.acos((up * up_ref).sum(-1).clamp(-1, 1))
        self._tilt = torch.acos(up[..., 2].clamp(-1, 1))
        self._z = pos[..., 2]
        self._body_rate = body_rate
        return TensorDict({
            "agents": {"observation": torch.nan_to_num(obs)},
            "stats": self.stats.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self) -> TensorDict:
        w = self.rw
        r = (w.get("pos", 1.0) * torch.exp(-self._pos_err / w.get("pos_scale", 0.1))
             + w.get("vel", 0.3) * torch.exp(-self._vel_err / w.get("vel_scale", 0.3))
             + w.get("att", 0.3) * torch.exp(-(self._tilt_err + self._yaw_err) / w.get("att_scale", 0.2))
             - w.get("body_rate", 0.01) * self._body_rate.square().sum(-1)
             - w.get("action_rate", 0.02) * (self.last_action - self.prev_action).square().sum(-1))

        failed = ((self._pos_err > self.term_pos_err) | (self._tilt > self.term_tilt) | (self._z < 0.15)).any(-1, True)
        ref_done = (self._ref_index() >= self.demo["len"][self.demo_idx] - 1).unsqueeze(-1)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) | ref_done
        terminated = failed
        r = torch.where(failed, r - w.get("fail", 10.0), r)

        self.stats["return"] += r.mean(-1, keepdim=True)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        k = self.progress_buf.unsqueeze(-1).clamp_min(1)
        self.stats["pos_error"].mul_((k - 1) / k).add_(self._pos_err.mean(-1, keepdim=True) / k)
        self.stats["yaw_error_deg"].mul_((k - 1) / k).add_(self._yaw_err.mean(-1, keepdim=True).rad2deg() / k)
        self.stats["ref_progress"][:] = (self._ref_index().float() / self.demo["len"][self.demo_idx]).unsqueeze(-1)
        self.stats["terminated_early"][:] = failed.float()
        return TensorDict({
            "agents": {"reward": r.unsqueeze(-1)},
            "done": terminated | truncated,
            "terminated": terminated,
            "truncated": truncated,
        }, self.batch_size)
