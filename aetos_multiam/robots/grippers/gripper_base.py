"""Gripper spec: data-only open/closed calibration (robosuite GripperModel analog).

Shaped like AM-bench's own ``GripperSpecCfg`` (ordered joints + open/closed
positions) since that's already a clean, validated pattern for this exact
problem -- a gripper here is just "which joints, and what values mean open
vs. closed", with ``format_action`` doing the robosuite-style mapping from a
single scalar action to per-joint targets.
"""

import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch

from aetos_multiam.assets import AssetNotAvailableError


@dataclass(frozen=True)
class GripperSpec:
    name: str
    joint_names: Sequence[str]
    open_joint_positions: Sequence[float]
    closed_joint_positions: Sequence[float]
    usd_path: Optional[str] = None
    """None when the gripper is always bundled into a manipulator's own USD."""
    root_body_name: Optional[str] = None
    """Root body to weld onto the manipulator's eef, when not bundled."""
    note: str = ""

    def __post_init__(self):
        if len(self.open_joint_positions) != len(self.joint_names):
            raise ValueError("open_joint_positions must have one value per joint.")
        if len(self.closed_joint_positions) != len(self.joint_names):
            raise ValueError("closed_joint_positions must have one value per joint.")

    def format_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map a per-env scalar action in [-1 (open), 1 (closed)], shape
        ``(num_envs,)``, to joint targets, shape ``(num_envs, num_joints)``."""
        t = ((action.clamp(-1.0, 1.0) + 1.0) / 2.0).unsqueeze(-1)  # (num_envs, 1)
        open_t = torch.as_tensor(self.open_joint_positions, dtype=t.dtype, device=t.device)
        closed_t = torch.as_tensor(self.closed_joint_positions, dtype=t.dtype, device=t.device)
        return open_t + t * (closed_t - open_t)

    def check_asset(self) -> None:
        if self.usd_path is not None and not osp.isfile(self.usd_path):
            raise AssetNotAvailableError("gripper:" + self.name, self.usd_path, self.note)


GRIPPER_REGISTRY: Dict[str, GripperSpec] = {}


def register_gripper(spec: GripperSpec) -> GripperSpec:
    if spec.name in GRIPPER_REGISTRY:
        raise ValueError(f"Gripper '{spec.name}' is already registered.")
    GRIPPER_REGISTRY[spec.name] = spec
    return spec
