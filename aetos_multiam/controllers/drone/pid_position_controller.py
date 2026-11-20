"""PID position control for any airframe here: an integral term on top of the
position loop the project already flies.

``LeePositionController`` and ``FullyActuatedPositionController`` share one
position law,

    acc = Kp * (pos - target_pos) + Kd * (vel - target_vel) - g - target_acc

which is a PD loop: nothing in it remembers an error. That is fine for a bare
airframe and wrong for an aerial manipulator doing work, where the drone is
pushed by things its model does not contain -- a door's hinge resisting a
pull, the static moment of an arm held out in front, a payload. A PD loop
answers a constant push with a constant offset; holding a gripper on a 46 mm
handle bar, a 3 cm offset is the difference between a grasp and a miss.

``PIDPositionController`` wraps either of those controllers and adds the
missing term. It needs no access to their internals, because the law above
already has a slot for it: ``target_acc`` enters ``acc`` exactly the way an
integral term would, so

    target_acc' = target_acc + Ki * integral(target_pos - pos) dt

turns the wrapped PD loop into a PID one, with Kp and Kd still the tuned
gains from the controller's own yaml (or overridden here), and with whatever
the wrapped controller does downstream of ``acc`` -- attitude from thrust
direction for an underactuated quad, a 6-DOF wrench allocation for the
fully-actuated hexa -- left exactly as it was. The attitude loop stays the
geometric PD it is in both; this is a PID on position, which is where the
disturbances of manipulation show up.

Two guards keep the integrator from doing harm:

* **Clamped** to ``integral_limit`` (anti-windup). The limit is stated as the
  largest acceleration the term may ever command, ``Ki * integral <=
  max_integral_acc``, because that is the number that has a physical meaning:
  it bounds the disturbance force the term can cancel at ``mass *
  max_integral_acc``.
* **Only accumulates near the setpoint** (``integration_radius``). A robot a
  metre from its target is in transit, not being disturbed; integrating there
  just stores up an overshoot for arrival.

Which frame the error is accumulated in is a choice (``integral_frame``),
because an integrator can only cancel a push that holds still in the frame it
remembers things in:

* ``"world"`` (default) suits wind, or a payload's weight -- pushes that keep
  their direction whichever way the drone is pointing.
* ``"heading"`` (world rotated by the drone's yaw) suits what a manipulator
  meets: a force that comes in through the arm and turns with the robot. A
  hexacopter swinging a door open is pulled back along the handle's path, and
  that direction rotates 90 deg during the swing. Measured with a world-frame
  integrator: what it had learned at the start of the swing was pointing
  sideways by the middle of it, and pushed the gripper 29 mm off the handle's
  arc. In the heading frame the same push is constant, and is cancelled.
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = ["PIDPositionController", "make_pid_position_controller"]

Gain = Union[float, Sequence[float]]


def _vec3(value: Gain) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.float32)
    return t.expand(3).clone() if t.dim() == 0 else t.reshape(3).clone()


class PIDPositionController(nn.Module):
    """``compute(...)`` has the wrapped controller's signature and return value.

    Args:
        inner: a ``LeePositionController`` or ``FullyActuatedPositionController``.
        dt: seconds between ``compute`` calls -- the integrator's step.
        ki: integral gain per axis, 1/s^2 (acceleration per metre-second of
            accumulated error).
        kp, kd: override the wrapped controller's own position / velocity
            gains; ``None`` keeps what its yaml carries.
        max_integral_acc: cap on ``ki * integral``, m/s^2, per axis.
        integration_radius: the integrator only runs while the position error
            is inside this distance, metres.
        integral_frame: ``"world"`` or ``"heading"`` -- see the module docstring.
        num_envs: how many robots one call flies.
    """

    def __init__(
        self,
        inner: nn.Module,
        dt: float,
        ki: Gain = (1.5, 1.5, 2.0),
        kp: Optional[Gain] = None,
        kd: Optional[Gain] = None,
        max_integral_acc: Gain = (2.0, 2.0, 3.0),
        integration_radius: float = 0.35,
        integral_frame: str = "world",
        num_envs: int = 1,
    ):
        super().__init__()
        self.inner = inner
        self.dt = float(dt)
        if kp is not None:
            inner.pos_gain.data.copy_(_vec3(kp).to(inner.pos_gain.device))
        if kd is not None:
            inner.vel_gain.data.copy_(_vec3(kd).to(inner.vel_gain.device))
        self.ki = nn.Parameter(_vec3(ki), requires_grad=False)
        self.max_integral_acc = nn.Parameter(_vec3(max_integral_acc), requires_grad=False)
        self.integration_radius = float(integration_radius)
        if integral_frame not in ("world", "heading"):
            raise ValueError(f"integral_frame must be 'world' or 'heading', got {integral_frame!r}")
        self.integral_frame = integral_frame
        #: Accumulated error, in ``integral_frame``.
        self.register_buffer("integral", torch.zeros(num_envs, 3))
        #: The I term as last commanded, m/s^2, world frame.
        self.register_buffer("integral_acc", torch.zeros(num_envs, 3))

    # The wrapped controller's own parameters, for callers that read them
    # (a payload feedforward raises ``mass``, for instance).
    @property
    def mass(self):
        return self.inner.mass

    @property
    def kp(self) -> torch.Tensor:
        return self.inner.pos_gain

    @property
    def kd(self) -> torch.Tensor:
        return self.inner.vel_gain

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.integral.zero_()
            self.integral_acc.zero_()
        else:
            self.integral[env_ids] = 0.0
            self.integral_acc[env_ids] = 0.0

    @staticmethod
    def _rotate_z(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=-1)

    def compute(
        self,
        root_state: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
        target_acc: Optional[torch.Tensor] = None,
        target_yaw: Optional[torch.Tensor] = None,
        body_rate: bool = False,
    ) -> torch.Tensor:
        if target_pos is not None:
            pos = root_state[..., :3].reshape(-1, 3)
            error = target_pos.expand(root_state.shape[:-1] + (3,)).reshape(-1, 3) - pos
            near = (error.norm(dim=-1, keepdim=True) < self.integration_radius).float()
            limit = self.max_integral_acc / self.ki.clamp_min(1e-9)
            if self.integral_frame == "heading":
                w, x, y, z = root_state[..., 3:7].reshape(-1, 4).unbind(-1)
                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                error = self._rotate_z(error, -yaw)
            self.integral = torch.maximum(
                torch.minimum(self.integral + near * error * self.dt, limit), -limit)
            self.integral_acc = self.ki * self.integral
            if self.integral_frame == "heading":
                self.integral_acc = self._rotate_z(self.integral_acc, yaw)
            feedforward = self.integral_acc.reshape(root_state.shape[:-1] + (3,))
            target_acc = feedforward if target_acc is None else target_acc + feedforward
        return self.inner.compute(
            root_state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_acc=target_acc,
            target_yaw=target_yaw,
            body_rate=body_rate,
        )

    forward = compute


def make_pid_position_controller(drone, dt: float, g: float = 9.81, mode: str = "auto",
                                 **pid_kwargs) -> PIDPositionController:
    """A PID position controller for ``drone``: ``make_position_controller``'s
    choice of inner loop (6-DOF allocation for a fully-actuated airframe, Lee
    otherwise) with the integral term added. ``pid_kwargs`` go to
    ``PIDPositionController``."""
    from .fully_actuated_controller import make_position_controller

    return PIDPositionController(make_position_controller(drone, g, mode), dt, **pid_kwargs)
