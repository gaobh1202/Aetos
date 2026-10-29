"""AerialManipulatorController -- merges drone/arm/gripper part controllers
into one flat action space (robosuite ``CompositeController`` analog).

A policy sees one action vector of size ``num_rotors + num_arm_joints + 1``
(the trailing 1 is the gripper's scalar open/close command, omitted if
there's no gripper); ``run`` slices it and forwards each piece to the
matching part controller, same as robosuite's ``CompositeController.set_goal``
splitting a flat action by ``_action_split_indexes``.

The drone part is raw per-rotor passthrough in this first pass (matching
this repo's own ``Hover`` task's action space) -- ``drone_controller`` is
accepted for forward-compatibility with a future cartesian/position
controller (e.g. wrapping ``LeePositionController``) but isn't invoked yet.
"""

from typing import Optional

import torch

from .arm.joint_position_controller import JointPositionController
from .gripper.binary_gripper_controller import BinaryGripperController


class AerialManipulatorController:
    def __init__(
        self,
        drone_controller: Optional[object] = None,
        arm_controller: Optional[JointPositionController] = None,
        gripper_controller: Optional[BinaryGripperController] = None,
    ):
        self.drone_controller = drone_controller
        self.arm_controller = arm_controller
        self.gripper_controller = gripper_controller

        self.num_arm_joints = (
            arm_controller.joint_lower.numel() if arm_controller is not None else 0
        )
        self.has_gripper = gripper_controller is not None

    @property
    def action_dim(self) -> int:
        # drone part is handled by the drone's own action_spec (variable size,
        # e.g. num_rotors or a 4-dim rate-controller command); only the
        # arm/gripper tail is fixed-size here.
        return self.num_arm_joints + (1 if self.has_gripper else 0)

    def split_action(self, action: torch.Tensor):
        """Slice a flat ``[..., drone_dim + num_arm_joints + has_gripper]`` action."""
        drone_dim = action.shape[-1] - self.action_dim
        drone_action = action[..., :drone_dim]
        rest = action[..., drone_dim:]

        arm_action = rest[..., : self.num_arm_joints] if self.num_arm_joints else None
        gripper_action = rest[..., -1] if self.has_gripper else None
        return drone_action, arm_action, gripper_action

    def run(self, drone, action: torch.Tensor):
        drone_action, arm_action, gripper_action = self.split_action(action)

        arm_targets = (
            self.arm_controller.compute(arm_action) if arm_action is not None else None
        )
        gripper_targets = (
            self.gripper_controller.compute(gripper_action)
            if gripper_action is not None
            else None
        )
        return drone.apply_action(drone_action, arm_targets, gripper_targets)
