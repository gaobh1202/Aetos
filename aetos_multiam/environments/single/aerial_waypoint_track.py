"""AerialWaypointTrack -- learn the DRONE controller (CTBR) while the arm moves, fully coupled.

Stage 1 of plan.md. One ``iris650_soarm100`` per env:

* action: PX4's offboard interface, body rates + collective thrust (as ``AerialTransportTrack``),
  closed by ``PX4RateController`` every physics substep;
* reference: a random waypoint route per episode -- dwell at a waypoint, move to the next one on
  a smooth (quintic) profile -- or a pure hover;
* the arm is NOT learned. Its joint targets come from one of three sources, drawn per episode:
  ``dataset`` replays a recorded grasp demo of the arm policy's training set (random demo, random
  speed, played back and forth), ``random`` wanders between random poses, ``static`` holds one;
* the arm's dynamics are NOT masked (``arm_dynamics_residual: 1.0``): every reaction force and
  torque of the arm reaches the airframe;
* a payload: a downward force at the gripper switched on at a random time (grasping something).

The policy sees what the arm is about to do: besides the arm's joint positions, the observation
carries the arm's joint targets now and at ``ARM_LOOKAHEAD`` steps ahead -- the action chunk an
arm policy such as ACT has already committed to. ``task.arm_obs`` cuts this down for ablations:
``preview`` (all of it), ``state`` (joint positions and the current target), ``none``.
"""

import math
import os
from typing import List, Sequence

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from omni_drones.utils.torch import (
    euler_to_quaternion, quat_rotate, quat_rotate_inverse, quaternion_to_rotation_matrix,
)
from omni_drones.utils.torchrl import AgentSpec

from aetos_multiam.controllers.drone.px4_rate_controller import ctbr_from_lee, denormalize_ctbr, normalize_ctbr
from aetos_multiam.scenes import Arena
from aetos_multiam.utils import freeze_arm_dynamics

from ..collaborative.collaborative_base import CollaborativeAerialEnv

#: Reference offsets (control steps) shown to the policy.
LOOKAHEAD = (0, 8, 16, 32)
#: Arm joint-target offsets (control steps) shown to the policy: about 1 s at 62.5 Hz.
ARM_LOOKAHEAD = (8, 16, 32, 64)
G = 9.81
GRASP_LOCAL = (0.0108, -0.0004, -0.0682)  # gripper centre on the 'gripper' link
ARM_MODES = ("dataset", "random", "static")


# -- the env's observation and reward as pure, differentiable functions ---------------------------
# The env itself calls them, and so does a learned world model that imagines rollouts
# (scripts/train_aerial_waypoint_wm.py), so the two cannot drift apart.
#
# Physical state s (..., STATE_DIM), env frame: position, rotation matrix (row-major), linear velocity,
# body rate, arm joint positions and velocities, rotor throttles, the rate loop's integrator.
S_POS, S_ROT, S_VEL, S_RATE = slice(0, 3), slice(3, 12), slice(12, 15), slice(15, 18)
S_JP, S_JV, S_THR, S_INT = slice(18, 24), slice(24, 30), slice(30, 34), slice(34, 37)
STATE_DIM = 37
# Exogenous signals per control step (..., EXO_DIM): what is known ahead of time and not the drone's doing.
X_REF_POS, X_REF_VEL, X_ARM, X_LOAD = slice(0, 3), slice(3, 6), slice(6, 12), 12
EXO_DIM = 13


def _norm(x: torch.Tensor) -> torch.Tensor:
    return (x.square().sum(-1) + 1e-12).sqrt()  # a norm whose gradient exists at 0


def _heading(s: torch.Tensor):
    R = s[..., S_ROT].reshape(*s.shape[:-1], 3, 3)
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    c, sn = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)

    def to_h(v):  # world -> heading frame (yaw removed)
        return torch.cat([c * v[..., 0:1] + sn * v[..., 1:2], -sn * v[..., 0:1] + c * v[..., 1:2], v[..., 2:3]], -1)

    return R, yaw, to_h


def policy_obs(s, last_action, exo, ref_yaw, arm_obs: str = "preview") -> torch.Tensor:
    """Observation at state s. exo (..., >= 1 + max lookahead, EXO_DIM) starts at the current control step."""
    R, yaw, to_h = _heading(s)
    Rh = torch.stack([to_h(R[..., :, i]) for i in range(3)], -1).reshape(*s.shape[:-1], 9)
    rel = [to_h(exo[..., k, X_REF_POS] - s[..., S_POS]) for k in LOOKAHEAD]
    yaw_err = ref_yaw - yaw
    jp, jv, tgt = s[..., S_JP], s[..., S_JV], exo[..., 0, X_ARM]
    zero = torch.zeros_like(jp)
    if arm_obs == "none":
        arm = [zero] * (3 + len(ARM_LOOKAHEAD))
    elif arm_obs == "state":
        arm = [jp, jv * 0.2, tgt - jp] + [zero] * len(ARM_LOOKAHEAD)
    else:  # the committed chunk, relative to now
        arm = [jp, jv * 0.2, tgt - jp] + [exo[..., k, X_ARM] - tgt for k in ARM_LOOKAHEAD]
    return torch.cat([Rh, s[..., S_RATE], to_h(s[..., S_VEL]), *rel, to_h(exo[..., 0, X_REF_VEL]),
                      yaw_err.sin().unsqueeze(-1), yaw_err.cos().unsqueeze(-1), last_action, *arm], -1)


def tracking_errors(s, exo0, ref_yaw):
    """(position error m, velocity error m/s, yaw error rad, tilt rad, height m) of state s against exo0 (..., EXO_DIM)."""
    R, yaw, _ = _heading(s)
    pos_err = _norm(exo0[..., X_REF_POS] - s[..., S_POS])
    vel_err = _norm(s[..., S_VEL] - exo0[..., X_REF_VEL])
    d = ref_yaw - yaw
    yaw_err = torch.atan2(d.sin(), d.cos()).abs()
    tilt = torch.atan2(_norm(R[..., :2, 2]), R[..., 2, 2])
    return pos_err, vel_err, yaw_err, tilt, s[..., 2]


def tracking_reward(s, action, prev_action, exo0, ref_yaw, w: dict) -> torch.Tensor:
    """Reward for arriving in state s with `action` (the one before it: `prev_action`); no failure penalty."""
    pos_err, vel_err, yaw_err, tilt, _ = tracking_errors(s, exo0, ref_yaw)
    return (w.get("pos", 1.0) * torch.exp(-pos_err / w.get("pos_scale", 0.1))
            + w.get("vel", 0.3) * torch.exp(-vel_err / w.get("vel_scale", 0.3))
            + w.get("att", 0.3) * torch.exp(-(tilt + yaw_err) / w.get("att_scale", 0.2))
            - w.get("body_rate", 0.01) * s[..., S_RATE].square().sum(-1)
            - w.get("action_rate", 0.02) * (action - prev_action).square().sum(-1))


class AerialWaypointTrack(CollaborativeAerialEnv):
    def __init__(self, cfg, headless):
        t = cfg.task
        if t.get("drone_action", "rotor") != "ctbr" or len(t.aerial_manipulators) != 1:
            raise ValueError("AerialWaypointTrack needs task.drone_action=ctbr and exactly one aerial manipulator.")
        self.arm_residual = float(t.get("arm_dynamics_residual", 1.0))
        self.arm_obs = str(t.get("arm_obs", "preview"))
        assert self.arm_obs in ("preview", "state", "none"), self.arm_obs
        self.arm_mode_probs = [float(t.arm_motion.get(m, 0.0)) for m in ARM_MODES]
        self.arm_speed = tuple(t.arm_motion.get("dataset_speed", (0.6, 1.6)))
        self.arm_random_speed = float(t.arm_motion.get("random_max_speed", 1.5))
        self.arm_random_interval = tuple(t.arm_motion.get("random_interval_s", (0.4, 2.0)))
        self.payload_max = float(t.get("payload_max_n", 0.0))
        self.payload_prob = float(t.get("payload_prob", 0.5))
        self.route = dict(t.route)
        self.reset_pos_noise = float(t.get("reset_pos_noise_m", 0.05))
        self.reset_rot_noise = math.radians(float(t.get("reset_rot_noise_deg", 5.0)))
        self.term_pos_err = float(t.get("terminate_pos_error_m", 1.0))
        self.term_tilt = math.radians(float(t.get("terminate_tilt_deg", 60.0)))
        self.rw = dict(t.get("reward", {}))
        bank = t.arm_motion.bank
        bank = bank if os.path.isabs(bank) else os.path.join(os.path.dirname(__file__), "..", "..", "..", bank)
        z = np.load(bank)
        self._bank = torch.as_tensor(z["action"], dtype=torch.float32)  # (E, T, 6)
        self._bank_len = torch.as_tensor(z["length"])
        self._bank_fps = float(z["fps"])
        # UJITSO's cooked-mesh service deadlocks PhysX start-up at >= 8 envs (gripper convex decomposition).
        import carb
        carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
        super().__init__(cfg, headless)

    # -- scene ---------------------------------------------------------------

    def _build_arena(self):
        return Arena()

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        return (0.0, 0.0, 1.5)

    def _load_model(self):
        paths = super()._load_model()
        if not self.cfg.task.get("visuals", False):
            # Headless training: invisible prims are not loaded by the RTX renderer (RAM, build time).
            from pxr import UsdGeom
            import isaacsim.core.utils.prims as prim_utils
            UsdGeom.Imageable(prim_utils.get_prim_at_path(self.template_env_ns)).MakeInvisible()
        return paths

    # -- setup ---------------------------------------------------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        N, dev = self.num_envs, self.device
        am = self.aerial_manipulators[0]
        self.am, view = am, am.drone._view
        self.arm_idx = am.teleop_joint_indices
        self.n_arm = len(self.arm_idx)
        if self.arm_residual < 1.0:
            freeze_arm_dynamics(am, residual=self.arm_residual)
        # shoulder_pan's and the jaw's authored drives are tuned for a table-mounted arm (see AerialTransportTrack).
        pan = self.cfg.task.get("shoulder_pan_gains", None)
        if pan:
            view.set_gains(kps=torch.full((N, 1), float(pan[0]), device=dev), kds=torch.full((N, 1), float(pan[1]), device=dev),
                           joint_indices=self.arm_idx[0:1])
        view.set_gains(kps=torch.full((N, 1), 10.0, device=dev), kds=torch.full((N, 1), 0.5, device=dev),
                       joint_indices=self.arm_idx[5:6])
        lim = view.get_dof_limits().reshape(-1, view.num_dof, 2)[0][self.arm_idx.to(view.get_dof_limits().device)].to(dev)
        self._bank, self._bank_len = self._bank.to(dev), self._bank_len.to(dev)
        # Random arm poses stay inside what the demos visited: joint limits alone would let the arm reach the rotors.
        lo = torch.maximum(self._bank.amin((0, 1)), lim[:, 0])
        hi = torch.minimum(self._bank.amax((0, 1)), lim[:, 1])
        self.arm_lo, self.arm_hi = lo, hi

        self.ctrl_dt = self.dt * self.substeps
        self.T = int(self.max_episode_length) + max(max(LOOKAHEAD), max(ARM_LOOKAHEAD)) + 2
        self.ref_pos = torch.zeros(N, self.T, 3, device=dev)
        self.ref_vel = torch.zeros(N, self.T, 3, device=dev)
        self.ref_yaw = torch.zeros(N, device=dev)
        self.arm_seq = torch.zeros(N, self.T, self.n_arm, device=dev)
        self.arm_mode = torch.zeros(N, dtype=torch.long, device=dev)
        self.load_n = torch.zeros(N, device=dev)       # payload force, N
        self.load_on = torch.zeros(N, dtype=torch.long, device=dev)  # control step it switches on
        self._load = torch.zeros(N, 3, device=dev)
        self.last_action = torch.zeros(N, 1, 4, device=dev)
        self.prev_action = torch.zeros_like(self.last_action)

        self.lee = None
        self.total_mass = float(am.drone.params["mass"])
        self.hover_frac = self.total_mass * G / self.rate_controllers[0].max_total_thrust

        # The payload enters through the one force write apply_action makes (a second call would replace it).
        orig = am.drone.base_link.apply_forces_and_torques_at_pos
        L, g = view.num_bodies, list(view.body_names).index("gripper")
        com_g = view.get_body_coms()[0].reshape(-1, L, 3)[:, g].to(dev)
        grasp_local = torch.tensor([GRASP_LOCAL], device=dev).expand(N, 3)
        rows = torch.arange(N, device=dev) * L

        def patched(forces=None, torques=None, positions=None, is_global=True):
            if forces is None or self.payload_max <= 0.0:
                return orig(forces, torques, positions, is_global=is_global)
            lt = view._physics_view.get_link_transforms().reshape(-1, L, 7)[:, g]
            q = torch.cat([lt[:, 6:7], lt[:, 3:6]], -1)  # PhysX xyzw -> wxyz
            arm = quat_rotate(q, grasp_local) - quat_rotate(q, com_g)  # link CoM -> gripper centre, world
            f = torch.zeros(N * L, 3, device=dev)
            tq = torch.zeros_like(f)
            f[rows], tq[rows] = forces.reshape(N, 3), torques.reshape(N, 3)  # root link is body 0
            f[rows + g], tq[rows + g] = self._load, torch.cross(arm, self._load, dim=-1)
            view._physics_view.apply_forces_and_torques_at_position(
                force_data=f, torque_data=tq, position_data=None, indices=torch.arange(N, device=dev), is_global=True)

        am.drone.base_link.apply_forces_and_torques_at_pos = patched

    def _obs_dim(self) -> int:
        return 9 + 3 + 3 + 3 * len(LOOKAHEAD) + 3 + 2 + 4 + self.n_arm * (3 + len(ARM_LOOKAHEAD))

    def _setup_observables(self) -> None:
        self.n_arm = 6  # so_arm100: 5 arm joints + jaw (checked against the live view in _setup_references)
        self.obs_dim = self._obs_dim()
        stats = ("return", "episode_len", "pos_error", "yaw_error_deg", "tilt_deg", "ref_progress", "terminated_early")
        self.observation_spec = Composite({
            "agents": {"observation": Unbounded((1, self.obs_dim))},
            "wm": {"state": Unbounded(STATE_DIM)},  # physical state, for a learned world model
            "stats": {k: Unbounded(1) for k in stats},
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({"agents": {"action": Bounded(-1, 1, (1, 4))}}).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({"agents": {"reward": Unbounded((1, 1))}}).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec("drone", 1, observation_key=("agents", "observation"),
                                             action_key=("agents", "action"), reward_key=("agents", "reward"))
        self.stats = self.observation_spec["stats"].zero()

    # -- per-episode reference and arm motion ----------------------------------

    def _make_route(self, n: int):
        """Waypoint route for n envs: (pos (n, T, 3), vel (n, T, 3)) in the env frame."""
        dev, r = self.device, self.route
        K = int(r.get("waypoints", 6))
        lo = torch.tensor(r.get("box_lo", (-1.5, -1.5, 1.0)), device=dev)
        hi = torch.tensor(r.get("box_hi", (1.5, 1.5, 2.2)), device=dev)
        step = torch.tensor(r.get("max_step", (1.5, 1.5, 0.5)), device=dev)
        hover = torch.rand(n, device=dev) < float(r.get("hover_prob", 0.3))
        wp = torch.zeros(n, K + 1, 3, device=dev)
        wp[:, 0] = lo + (hi - lo) * torch.rand(n, 3, device=dev)
        for k in range(K):
            d = (torch.rand(n, 3, device=dev) * 2 - 1) * step
            d[hover] = 0.0
            wp[:, k + 1] = torch.minimum(torch.maximum(wp[:, k] + d, lo), hi)
        dwell_lo, dwell_hi = r.get("dwell_s", (0.5, 3.0))
        v_lo, v_hi = r.get("speed", (0.3, 1.2))
        dwell = dwell_lo + (dwell_hi - dwell_lo) * torch.rand(n, K, device=dev)
        speed = v_lo + (v_hi - v_lo) * torch.rand(n, K, device=dev)
        # A quintic's peak speed is 1.875 x its mean.
        move = ((wp[:, 1:] - wp[:, :-1]).norm(dim=-1) * 1.875 / speed).clamp_min(0.5)
        seg = dwell + move
        end = seg.cumsum(-1)
        t = torch.arange(self.T, device=dev).float().unsqueeze(0).expand(n, -1) * self.ctrl_dt
        k = torch.searchsorted(end, t.contiguous()).clamp_max(K - 1)
        start = torch.gather(end - seg, 1, k)
        s = ((t - start - torch.gather(dwell, 1, k)) / torch.gather(move, 1, k)).clamp(0.0, 1.0)
        blend = (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5).unsqueeze(-1)
        a = torch.gather(wp, 1, k.unsqueeze(-1).expand(-1, -1, 3))
        b = torch.gather(wp, 1, (k + 1).unsqueeze(-1).expand(-1, -1, 3))
        pos = a + (b - a) * blend
        vel = torch.zeros_like(pos)
        vel[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / self.ctrl_dt
        return pos, vel

    def _make_arm_motion(self, n: int):
        """Arm joint targets for n envs: (seq (n, T, n_arm), mode (n,))."""
        dev, T = self.device, self.T
        mode = torch.multinomial(torch.tensor(self.arm_mode_probs, device=dev), n, replacement=True)
        t = torch.arange(T, device=dev).float().unsqueeze(0).expand(n, -1)

        # dataset: a recorded demo, at a random speed, played back and forth from a random point.
        e = torch.randint(0, len(self._bank), (n,), device=dev)
        last = (self._bank_len[e] - 1).float().unsqueeze(-1)
        s_lo, s_hi = self.arm_speed
        rate = (s_lo + (s_hi - s_lo) * torch.rand(n, 1, device=dev)) * self._bank_fps * self.ctrl_dt
        u = torch.rand(n, 1, device=dev) * 2 * last + t * rate
        u = last - (u % (2 * last) - last).abs()  # triangle wave over [0, last]
        i0 = u.floor().long().clamp(0, self._bank.shape[1] - 2)
        w = (u - i0).unsqueeze(-1)
        demo = self._bank[e]  # (n, Tb, 6)
        g0 = torch.gather(demo, 1, i0.unsqueeze(-1).expand(-1, -1, self.n_arm))
        g1 = torch.gather(demo, 1, (i0 + 1).unsqueeze(-1).expand(-1, -1, self.n_arm))
        dataset = g0 + (g1 - g0) * w

        # random: a walk between random poses, each leg a cosine blend, bounded in speed.
        i_lo, i_hi = self.arm_random_interval
        interval = i_lo + (i_hi - i_lo) * torch.rand(n, 1, device=dev)
        legs = int(T * self.ctrl_dt / i_lo) + 2
        span = self.arm_hi - self.arm_lo
        knots = torch.zeros(n, legs + 1, self.n_arm, device=dev)
        knots[:, 0] = self.arm_lo + span * torch.rand(n, self.n_arm, device=dev)
        for k in range(legs):
            d = (torch.rand(n, self.n_arm, device=dev) * 2 - 1) * self.arm_random_speed * interval
            knots[:, k + 1] = torch.minimum(torch.maximum(knots[:, k] + d, self.arm_lo), self.arm_hi)
        ph = t * self.ctrl_dt / interval
        k = ph.floor().long().clamp_max(legs - 1)
        c = (0.5 - 0.5 * torch.cos(math.pi * (ph - k).clamp(0, 1))).unsqueeze(-1)
        a = torch.gather(knots, 1, k.unsqueeze(-1).expand(-1, -1, self.n_arm))
        b = torch.gather(knots, 1, (k + 1).unsqueeze(-1).expand(-1, -1, self.n_arm))
        random = a + (b - a) * c

        static = knots[:, :1].expand(-1, T, -1)
        seq = torch.where((mode == 0).reshape(n, 1, 1), dataset, torch.where((mode == 1).reshape(n, 1, 1), random, static))
        return seq, mode

    def _at(self, x: torch.Tensor, offset: int = 0, env_ids=None) -> torch.Tensor:
        """x (N, T, ...) at each env's current step + offset."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        t = (self.progress_buf[ids].long() + offset).clamp_max(self.T - 1)
        return x[ids, t]

    # -- reset ---------------------------------------------------------------

    def _reset(self, tensordict, **kwargs):
        if not getattr(self, "_warmed_up", False):
            # The first state write after scene creation is only partly applied by PhysX; one throwaway step.
            self._warmed_up = True
            super()._reset(None)
            self.sim.step(False)
        return super()._reset(tensordict, **kwargs)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n, dev = len(env_ids), self.device
        self.progress_buf[env_ids] = 0
        self.ref_pos[env_ids], self.ref_vel[env_ids] = self._make_route(n)
        self.ref_yaw[env_ids] = (torch.rand(n, device=dev) * 2 - 1) * math.pi
        self.arm_seq[env_ids], self.arm_mode[env_ids] = self._make_arm_motion(n)
        has = torch.rand(n, device=dev) < self.payload_prob
        self.load_n[env_ids] = torch.rand(n, device=dev) * self.payload_max * has
        self.load_on[env_ids] = (torch.rand(n, device=dev) * self.max_episode_length).long()

        d = self.am.drone
        d._reset_idx(env_ids, train=True)
        pos = self.ref_pos[env_ids, 0] + self.reset_pos_noise * torch.randn(n, 3, device=dev)
        rpy = self.reset_rot_noise * torch.randn(n, 3, device=dev)
        rpy[:, 2] += self.ref_yaw[env_ids]
        d.set_world_poses((pos + self.envs_positions[env_ids]).unsqueeze(1), euler_to_quaternion(rpy).unsqueeze(1),
                          env_indices=env_ids)
        d.set_velocities(torch.zeros(n, 1, 6, device=dev), env_indices=env_ids)
        ndof = d._view.num_dof
        d._view.set_joint_velocities(torch.zeros(n, 1, ndof, device=dev), env_indices=env_ids,
                                     joint_indices=torch.arange(ndof, device=dev))
        d.throttle[env_ids] = math.sqrt(self.hover_frac)  # thrust is linear in throttle^2
        q0 = self.arm_seq[env_ids, 0].unsqueeze(1)
        d._view.set_joint_positions(q0, env_indices=env_ids, joint_indices=self.arm_idx)
        d._view.set_joint_position_targets(q0, env_indices=env_ids, joint_indices=self.arm_idx)
        self.rate_controllers[0].reset(env_ids)
        hover = torch.tensor([0.0, 0.0, 0.0, 2 * self.hover_frac - 1], device=dev)
        self.last_action[env_ids] = hover
        self.prev_action[env_ids] = hover
        self.stats[env_ids] = 0.0

    # -- step ----------------------------------------------------------------

    def _pre_sim_step(self, tensordict: TensorDict) -> None:
        action = tensordict[("agents", "action")].clamp(-1, 1)
        if not getattr(self, "_substep_started", False):  # once per control step
            self._substep_started = True
            self.prev_action = self.last_action.clone()
            self.last_action = action.clone()
            self.am.drone._view.set_joint_position_targets(self._at(self.arm_seq).unsqueeze(1), joint_indices=self.arm_idx)
            self._load[:, 2] = -self.load_n * (self.progress_buf >= self.load_on)
        rate_sp, thrust = denormalize_ctbr(action[:, 0])
        state = self.am.drone.get_state()[:, 0, :13]
        self.am.apply_action(self.rate_controllers[0](state, rate_sp, thrust).unsqueeze(1))

    def _post_sim_step(self, tensordict: TensorDict) -> None:
        self._substep_started = False

    def expert_action(self) -> torch.Tensor:
        """A cascaded position controller's action for the current state (N, 1, 4): the classical baseline.
        It knows the vehicle's total mass but nothing about where the arm is or what it carries."""
        if self.lee is None:
            from aetos_multiam.controllers import make_position_controller
            self.lee = make_position_controller(self.am.drone, G).to(self.device)
        state = self.am.drone.get_state()[:, 0, :13]  # positions are already relative to each env's origin
        rate_sp, thrust = ctbr_from_lee(self.lee, state, self._at(self.ref_pos), self._at(self.ref_vel),
                                        self.ref_yaw.unsqueeze(-1), self.rate_controllers[0].max_total_thrust)
        return normalize_ctbr(rate_sp, thrust).unsqueeze(1)

    # -- observation / reward ----------------------------------------------------

    def wm_state(self) -> torch.Tensor:
        """Physical state (N, STATE_DIM), see S_* above."""
        d = self.am.drone
        state = d.get_state()[:, 0, :13]  # positions are already relative to each env's origin
        rot = state[:, 3:7]
        jp = d._view.get_joint_positions()[:, 0][:, self.arm_idx]
        jv = d._view.get_joint_velocities()[:, 0][:, self.arm_idx]
        return torch.cat([state[:, :3], quaternion_to_rotation_matrix(rot).reshape(-1, 9), state[:, 7:10],
                          quat_rotate_inverse(rot, state[:, 10:13]), jp, jv, d.throttle[:, 0],
                          self.rate_controllers[0].integral], -1)

    def wm_exo(self, steps: int) -> torch.Tensor:
        """Exogenous signals (N, steps, EXO_DIM) from each env's current control step on: the reference, the arm's
        joint targets (its committed action chunk) and the payload force."""
        t = self.progress_buf.long().unsqueeze(-1) + torch.arange(steps, device=self.device)
        tc = t.clamp_max(self.T - 1)
        rows = torch.arange(self.num_envs, device=self.device).unsqueeze(-1)
        load = self.load_n.unsqueeze(-1) * (t >= self.load_on.unsqueeze(-1))
        return torch.cat([self.ref_pos[rows, tc], self.ref_vel[rows, tc], self.arm_seq[rows, tc], load.unsqueeze(-1)], -1)

    def _compute_state_and_obs(self) -> TensorDict:
        s = self.wm_state()
        self._bad = ~torch.isfinite(s).all(-1)
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        exo = self.wm_exo(max(max(LOOKAHEAD), max(ARM_LOOKAHEAD)) + 1)
        obs = policy_obs(s, self.last_action[:, 0], exo, self.ref_yaw, self.arm_obs)
        self._pos_err, self._vel_err, self._yaw_err, self._tilt, self._z = tracking_errors(s, exo[:, 0], self.ref_yaw)
        self._s, self._exo0 = s, exo[:, 0]
        return TensorDict({
            "agents": {"observation": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)},
            "wm": {"state": s},
            "stats": self.stats.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self) -> TensorDict:
        w = self.rw
        r = tracking_reward(self._s, self.last_action[:, 0], self.prev_action[:, 0], self._exo0, self.ref_yaw, w)
        r = torch.nan_to_num(r, nan=0.0)
        failed = ((self._pos_err > self.term_pos_err) | (self._tilt > self.term_tilt) | (self._z < 0.3) | self._bad).unsqueeze(-1)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        r = torch.where(failed.squeeze(-1), r - w.get("fail", 10.0), r).unsqueeze(-1)

        self.stats["return"] += r
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        k = self.progress_buf.unsqueeze(-1).clamp_min(1)
        for key, v in (("pos_error", self._pos_err), ("yaw_error_deg", self._yaw_err.rad2deg()), ("tilt_deg", self._tilt.rad2deg())):
            self.stats[key].mul_((k - 1) / k).add_(torch.nan_to_num(v).unsqueeze(-1) / k)
        self.stats["ref_progress"][:] = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        self.stats["terminated_early"][:] = failed.float()
        return TensorDict({
            "agents": {"reward": r.unsqueeze(-1)},
            "done": failed | truncated,
            "terminated": failed,
            "truncated": truncated,
        }, self.batch_size)
