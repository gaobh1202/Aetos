"""Binary open/close gripper controller (robosuite ``GripperModel.format_action`` analog)."""

import torch

from aetos_multiam.robots.grippers.gripper_base import GripperSpec


class BinaryGripperController:
    def __init__(self, gripper_spec: GripperSpec):
        self.gripper_spec = gripper_spec

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """``action`` shape ``(num_envs,)`` in [-1 (open), 1 (closed)] -> joint
        targets, shape ``(num_envs, num_joints)``."""
        return self.gripper_spec.format_action(action)
