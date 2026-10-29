"""Simple PD joint actuator helper for arm/gripper joints.

Isaac Sim's PhysX implicit-PD drive is already a position actuator; this
just sets its gains on an ``ArticulationView`` for a given set of joint
indices, the same call pattern this repo's own ``hover.py`` uses for its
arm-trajectory joints (``self.drone._view.set_gains(...)``).
"""

import torch


def set_joint_gains(
    view,
    joint_indices: torch.Tensor,
    stiffness: float,
    damping: float,
    device,
) -> None:
    count = view.count
    n = joint_indices.numel()
    stiffness_t = torch.full((count, n), float(stiffness), device=device)
    damping_t = torch.full((count, n), float(damping), device=device)
    view.set_gains(stiffness_t, damping_t, joint_indices=joint_indices)
