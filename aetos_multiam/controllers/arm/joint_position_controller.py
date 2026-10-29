"""Direct joint-target arm controller (robosuite ``JointPositionController`` analog).

Isaac Sim's PhysX implicit drive already does the PD tracking once gains are
set (see ``actuators.joint_actuator.set_joint_gains``) -- this class's job is
just to turn a normalized action in [-1, 1] per joint into an absolute joint
target within that joint's limits, the same "clip to range, scale, offset"
step robosuite's part controllers do before writing ``sim.data.ctrl``.

An IK-based alternative (cartesian eef commands -> joint targets) is real
follow-up work, not implemented here -- see ``docs/ARCHITECTURE.md``.
"""

from typing import Optional, Sequence

import torch


class JointPositionController:
    def __init__(
        self,
        joint_lower: Sequence[float],
        joint_upper: Sequence[float],
        device: Optional[torch.device] = None,
    ):
        self.joint_lower = torch.as_tensor(joint_lower, dtype=torch.float32, device=device)
        self.joint_upper = torch.as_tensor(joint_upper, dtype=torch.float32, device=device)

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """``action`` in [-1, 1] per joint -> absolute joint-position targets."""
        action = action.clamp(-1.0, 1.0)
        t = (action + 1.0) / 2.0
        return self.joint_lower + t * (self.joint_upper - self.joint_lower)
