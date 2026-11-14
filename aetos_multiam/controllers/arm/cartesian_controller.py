"""End-effector Cartesian control for any mounted arm (robosuite ``OSC_POSE`` / Isaac Lab
``DifferentialIKController`` analog).

Command: a world-frame pose (or position only) for a point on the arm's end-effector link.
Output: joint-position targets for the arm's PD drives. One step is resolved-rate inverse
kinematics with damped least squares, so it stays bounded near singularities and on arms with
fewer than six joints, where it tracks position first and orientation as far as ``rot_weight``
allows.

It is not torque-level operational-space control: every arm here is position-servoed on the
real hardware, so the same interface is built on joint-position targets.

The Jacobian comes from the simulator (``EndEffector``), so no per-arm kinematic model is
needed and it runs batched over parallel envs. The math (``CartesianController``,
``pose_error``) is pure torch.

Self-check (no simulator): ``python aetos_multiam/controllers/arm/cartesian_controller.py``
"""

from typing import Optional, Sequence, Tuple

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, wxyz."""
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat([aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)], -1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def pose_error(target_pos, target_quat, pos, quat) -> Tuple[torch.Tensor, torch.Tensor]:
    """(position error, orientation error as a world-frame rotation vector) from pose to target."""
    d = quat_mul(target_quat, quat * quat.new_tensor([1.0, -1.0, -1.0, -1.0]))  # target * quat^-1
    d = d * torch.where(d[..., :1] < 0, -1.0, 1.0)  # the shorter way round
    n = d[..., 1:].norm(dim=-1, keepdim=True)
    return target_pos - pos, d[..., 1:] / n.clamp_min(1e-9) * 2 * torch.atan2(n, d[..., :1])


class CartesianController:
    def __init__(self, kp: float = 4.0, kr: float = 4.0, rot_weight: float = 0.1, damping: float = 0.05,
                 max_joint_speed: float = 1.5, max_lead: float = 0.25, position_first: bool = False):
        """``kp``/``kr``: 1/s on position / orientation error. ``rot_weight`` (m/rad): how much a
        radian of orientation error counts against a metre of position error when the arm cannot
        satisfy both. ``max_lead`` (rad): the command may lead the measured joint by at most this
        much, so it does not wind up while a drive lags. ``position_first``: strict priority --
        position is tracked exactly and orientation only with the joints' leftover freedom
        (``rot_weight`` is then unused); for arms with fewer than six joints following a full pose."""
        self.kp, self.kr, self.rot_weight, self.damping = kp, kr, rot_weight, damping
        self.position_first = position_first
        self.max_joint_speed, self.max_lead = max_joint_speed, max_lead

    def step(self, jacobian: torch.Tensor, pos_err: torch.Tensor, rot_err: Optional[torch.Tensor],
             q: torch.Tensor, q_cmd: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor, dt: float) -> torch.Tensor:
        """New joint targets (N, n). ``jacobian``: (N, 6, n) at the controlled point, linear rows
        first. ``rot_err=None`` controls position only. ``q_cmd``: the previous targets."""
        def solve(J, e):  # damped least squares
            A = J @ J.transpose(1, 2) + self.damping ** 2 * torch.eye(J.shape[1], device=J.device, dtype=J.dtype)
            return (J.transpose(1, 2) @ torch.linalg.solve(A, e.unsqueeze(-1))).squeeze(-1)

        Jp, Jr = jacobian[:, :3], jacobian[:, 3:]
        if rot_err is None:
            qd = solve(Jp, self.kp * pos_err)
        elif self.position_first:
            qd = solve(Jp, self.kp * pos_err)
            # Joint motions that leave the position alone: I - pinv(Jp) Jp.
            N = torch.eye(Jp.shape[2], device=Jp.device, dtype=Jp.dtype) - torch.linalg.pinv(Jp) @ Jp
            q2 = (N @ solve(Jr @ N, self.kr * rot_err - (Jr @ qd.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
            # Scaled as a whole: clamping it joint by joint would bend it out of that null space.
            top = q2.abs().amax(-1, keepdim=True).clamp_min(1e-9)
            qd = qd + q2 * (0.5 * self.max_joint_speed / top).clamp(max=1.0)
        else:
            qd = solve(torch.cat([Jp, self.rot_weight * Jr], 1),
                       torch.cat([self.kp * pos_err, self.rot_weight * self.kr * rot_err], -1))
        qd = qd.clamp(-self.max_joint_speed, self.max_joint_speed)
        lead = self.max_lead
        return torch.minimum(torch.maximum(q_cmd + qd * dt, q - lead), q + lead).clamp(q_lo, q_hi)


class EndEffector:
    """Pose and Jacobian of a point on one link of a floating-base articulation, read from the
    simulator. ``view`` is the drone's ArticulationView (arm joints are part of it)."""

    def __init__(self, view, link_name: str, dof_idx: torch.Tensor, point_local: Sequence[float] = (0.0, 0.0, 0.0)):
        names = [n.split("/")[-1] for n in view.body_names]
        self.view, self.n_links, self.link = view, len(names), names.index(link_name.split("/")[-1])
        dev = dof_idx.device
        self.cols = 6 + dof_idx  # floating base: six base columns come first
        self.point_local = torch.as_tensor(point_local, dtype=torch.float32, device=dev)
        self.com_local = view.get_body_coms()[0].reshape(-1, self.n_links, 3)[0, self.link].to(dev)

    def _link(self):
        lt = self.view._physics_view.get_link_transforms().reshape(-1, self.n_links, 7)[:, self.link]
        return lt[:, :3], torch.cat([lt[:, 6:7], lt[:, 3:6]], -1)  # PhysX quats are xyzw

    def pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """World position of the controlled point and the link quaternion (wxyz)."""
        p, q = self._link()
        return p + quat_rotate(q, self.point_local.expand_as(p)), q

    def jacobian(self) -> torch.Tensor:
        """(N, 6, n): world linear velocity of the controlled point, then angular velocity."""
        J = self.view._physics_view.get_jacobians()[:, self.link][:, :, self.cols]
        p, q = self._link()
        # PhysX gives the linear rows at the link's centre of mass: shift them to the point.
        r = quat_rotate(q, (self.point_local - self.com_local).expand_as(p)).unsqueeze(-1)
        return torch.cat([J[:, :3] + torch.cross(J[:, 3:], r.expand_as(J[:, 3:]), dim=1), J[:, 3:]], 1)


if __name__ == "__main__":  # self-check on a planar 3R arm (analytic kinematics, no simulator)
    L = torch.tensor([0.3, 0.25, 0.15], dtype=torch.float64)

    def fk(q):
        a = q.cumsum(-1)
        return torch.stack([(L * a.cos()).sum(-1), (L * a.sin()).sum(-1), torch.zeros_like(a[:, 0])], -1), a[:, -1]

    def jac(q):
        J = torch.zeros(q.shape[0], 6, 3, dtype=torch.float64)
        a = q.cumsum(-1)
        for j in range(3):
            J[:, 0, j], J[:, 1, j] = -(L[j:] * a[:, j:].sin()).sum(-1), (L[j:] * a[:, j:].cos()).sum(-1)
        J[:, 5] = 1.0
        return J

    def yaw_quat(a):
        return torch.stack([(a / 2).cos(), torch.zeros_like(a), torch.zeros_like(a), (a / 2).sin()], -1)

    ctl = CartesianController(max_lead=10.0)
    q = torch.tensor([[0.3, 0.6, -0.4], [-0.5, 1.0, 0.5]], dtype=torch.float64)
    goal_p, goal_a = fk(q + torch.tensor([0.25, -0.3, 0.2], dtype=torch.float64))  # a reachable pose
    lim = torch.full((3,), 3.0, dtype=torch.float64)
    for _ in range(600):
        p, a = fk(q)
        e_p, e_r = pose_error(goal_p, yaw_quat(goal_a), p, yaw_quat(a))
        q = ctl.step(jac(q), e_p, e_r, q, q, -lim, lim, 0.016)  # ideal drives: joint follows the command
    p, a = fk(q)
    assert (p - goal_p).norm(dim=-1).max() < 1e-3 and (a - goal_a).abs().max() < 1e-2, (p - goal_p, a - goal_a)
    far = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64).expand(2, 3)  # out of reach: must stay bounded
    for _ in range(600):
        q = ctl.step(jac(q), far - fk(q)[0], None, q, q, -lim, lim, 0.016)
    assert torch.isfinite(q).all() and (fk(q)[0][:, 0] - L.sum()).abs().max() < 2e-2, fk(q)[0]
    # position_first: a reachable position with an orientation the arm cannot also reach.
    pf = CartesianController(max_lead=10.0, position_first=True)
    q = torch.tensor([[0.3, 0.6, -0.4]], dtype=torch.float64)
    goal_p = torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float64)  # far out along x
    goal_q = yaw_quat(torch.tensor([1.5], dtype=torch.float64))  # ... pointing along y
    for _ in range(1500):
        p, a = fk(q)
        e_p, e_r = pose_error(goal_p, goal_q, p, yaw_quat(a))
        q = pf.step(jac(q), e_p, e_r, q, q, -lim, lim, 0.016)
    assert (fk(q)[0] - goal_p).norm() < 2e-3, fk(q)[0]
    print("cartesian_controller self-check OK")
