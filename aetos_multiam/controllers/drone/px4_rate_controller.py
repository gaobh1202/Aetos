"""PX4-style collective-thrust + body-rate (CTBR) interface for the sim.

A policy that is meant to run on a real PX4 vehicle should output what PX4's
offboard mode accepts: three body rates and one normalized collective thrust
(MAVLink SET_ATTITUDE_TARGET with the attitude ignored). PX4 then closes the
rate loop and allocates to the motors itself. To train against that
interface, the sim needs the same split:

``PX4RateController``
    The part PX4 keeps: a per-axis body-rate PID (D on the measured rate,
    low-pass filtered; clamped integrator), run every PHYSICS substep, i.e.
    at the rate loop's own frequency, not the policy's -- then the existing
    OmniDrones allocation (``compute_parameters``) to per-rotor commands.
    Gains are in angular acceleration per unit error and are multiplied by
    the inertia inside the mixer, the same convention OmniDrones'
    ``RateController`` uses; PX4's own MC_*RATE_* values are in normalized
    torque and do not transfer numerically -- re-tune them on the real
    airframe, the structure (PID, D on measurement, integrator limit,
    thrust as a fraction of max) is what carries over.

``ctbr_from_lee``
    The part the policy replaces: the Lee position loop's desired
    acceleration and attitude, turned into a body-rate setpoint by a PX4-style
    attitude P controller (MC_ROLL_P/MC_PITCH_P/MC_YAW_P defaults) and a
    collective thrust fraction. Used as the scripted expert that labels demos
    in the policy's own action space.

Normalized action (what a policy emits, what gets stored as ``action``):
``[p, q, r] / max_body_rate`` and ``2 * thrust_frac - 1``, all in [-1, 1].
On PX4: ``thrust = (a[3] + 1) / 2``, ``rates = a[:3] * max_body_rate``.
"""

import math

import torch
from torch import nn

from omni_drones.controllers.lee_position_controller import compute_parameters
from omni_drones.utils.torch import normalize, quat_rotate_inverse, quaternion_to_rotation_matrix

#: PX4 defaults for the attitude P loop (MC_ROLL_P, MC_PITCH_P, MC_YAW_P) and
#: the rate limits (MC_ROLLRATE_MAX 220, MC_PITCHRATE_MAX 220, MC_YAWRATE_MAX 200 deg/s).
PX4_ATTITUDE_P = (6.5, 6.5, 2.8)
PX4_MAX_BODY_RATE_DEG = (220.0, 220.0, 200.0)

#: Rate-loop gains per airframe (angular acceleration units, see module doc).
#: iris_650: P keeps the same roll/pitch damping the Lee controller's rate
#: term had (1.96 / 2.59 N.m per rad/s over Ixx / Iyy = 14.3 1/s) raised to
#: 20 1/s for a ~50 ms rate time constant against the 28.5 ms motor lag; I
#: rejects the frozen arm's residual torque; D is small, PX4-style.
RATE_GAINS = {
    "iris_650": dict(p=(20.0, 20.0, 6.0), i=(10.0, 10.0, 2.0), d=(0.05, 0.05, 0.0),
                     int_lim=(8.0, 8.0, 3.0), d_cutoff_hz=30.0),
}


class PX4RateController(nn.Module):
    def __init__(self, uav_params: dict, dt: float, num_envs: int, device, gains: dict = None):
        super().__init__()
        name = uav_params["name"]
        g = gains or RATE_GAINS.get(name)
        if g is None:
            raise KeyError(f"No PX4 rate-loop gains for airframe {name!r}; add them to RATE_GAINS.")
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        I = torch.diag(torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1.0]))
        self.dt = float(dt)
        self.mixer = compute_parameters(rotor_config, I).float().to(device)
        max_thrusts = (torch.as_tensor(rotor_config["max_rotation_velocities"]).square()
                       * torch.as_tensor(rotor_config["force_constants"])).float().to(device)
        self.max_thrusts = max_thrusts
        self.max_total_thrust = float(max_thrusts.sum())
        self.kp = torch.tensor(g["p"], device=device)
        self.ki = torch.tensor(g["i"], device=device)
        self.kd = torch.tensor(g["d"], device=device)
        self.int_lim = torch.tensor(g["int_lim"], device=device)
        self.d_alpha = 1.0 - math.exp(-2 * math.pi * g["d_cutoff_hz"] * self.dt)
        self.integral = torch.zeros(num_envs, 3, device=device)
        self.prev_rate = torch.zeros(num_envs, 3, device=device)
        self.d_filt = torch.zeros(num_envs, 3, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        idx = slice(None) if env_ids is None else env_ids
        self.integral[idx] = 0.0
        self.d_filt[idx] = 0.0
        self.fresh[idx] = True

    def forward(self, root_state: torch.Tensor, rate_sp: torch.Tensor, thrust_frac: torch.Tensor):
        """root_state (N, 13) world frame; rate_sp (N, 3) body rad/s; thrust_frac (N, 1) in [0, 1].
        Returns per-rotor commands (N, num_rotors) in OmniDrones' [-1, 1] convention."""
        rot, angvel = root_state[:, 3:7], root_state[:, 10:13]
        rate = quat_rotate_inverse(rot, angvel)
        # No derivative kick on the first step after a reset.
        self.prev_rate[self.fresh] = rate[self.fresh]
        self.fresh[:] = False
        err = rate_sp - rate
        self.integral = torch.maximum(torch.minimum(self.integral + err * self.dt, self.int_lim / self.ki.clamp_min(1e-6)),
                                      -self.int_lim / self.ki.clamp_min(1e-6))
        d_raw = (rate - self.prev_rate) / self.dt
        self.prev_rate = rate
        self.d_filt = self.d_filt + self.d_alpha * (d_raw - self.d_filt)
        ang_acc = self.kp * err + self.ki * self.integral - self.kd * self.d_filt
        thrust = thrust_frac.clamp(0.0, 1.0) * self.max_total_thrust
        rotor_thrust = (self.mixer @ torch.cat([ang_acc, thrust], dim=-1).T).T
        return rotor_thrust / self.max_thrusts * 2 - 1


def ctbr_from_lee(lee, root_state, target_pos, target_vel, target_yaw, max_total_thrust,
                  att_p=PX4_ATTITUDE_P, max_rate_deg=PX4_MAX_BODY_RATE_DEG):
    """Lee position loop -> (body-rate setpoint (N, 3) rad/s, thrust fraction (N, 1)).

    Same desired acceleration and desired rotation as ``LeePositionController``
    (it reads that controller's gains, mass and gravity), but the attitude
    error is closed by a PX4-style P loop into body rates instead of torques.
    """
    pos, rot, vel = root_state[:, :3], root_state[:, 3:7], root_state[:, 7:10]
    acc = (pos - target_pos) * lee.pos_gain + (vel - target_vel) * lee.vel_gain - lee.g
    R = quaternion_to_rotation_matrix(rot)
    b1_des = torch.cat([torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)], dim=-1)
    b3_des = -normalize(acc)
    b2_des = normalize(torch.cross(b3_des, b1_des, 1))
    R_des = torch.stack([b2_des.cross(b3_des, 1), b2_des, b3_des], dim=-1)
    e = 0.5 * (torch.bmm(R_des.transpose(-2, -1), R) - torch.bmm(R.transpose(-2, -1), R_des))
    e_R = torch.stack([e[:, 2, 1], e[:, 0, 2], e[:, 1, 0]], dim=-1)
    max_rate = torch.tensor(max_rate_deg, device=rot.device).deg2rad()
    rate_sp = (-torch.tensor(att_p, device=rot.device) * e_R).clamp(-max_rate, max_rate)
    thrust = -lee.mass * (acc * R[:, :, 2]).sum(-1, True)
    return rate_sp, (thrust / max_total_thrust).clamp(0.0, 1.0)


def normalize_ctbr(rate_sp, thrust_frac, max_rate_deg=PX4_MAX_BODY_RATE_DEG):
    max_rate = torch.tensor(max_rate_deg, device=rate_sp.device).deg2rad()
    return torch.cat([rate_sp / max_rate, thrust_frac * 2 - 1], dim=-1)


def denormalize_ctbr(action, max_rate_deg=PX4_MAX_BODY_RATE_DEG):
    max_rate = torch.tensor(max_rate_deg, device=action.device).deg2rad()
    return action[..., :3] * max_rate, (action[..., 3:4] + 1) / 2
