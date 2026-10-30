"""AerialManipulatorBase -- runtime composition of drone + manipulator + gripper.

This is the robosuite ``Robot`` wrapper's analog: at *spawn* time it combines
a drone model, an optional manipulator model, and an optional gripper model
(each just a data spec until now) into one physical prim tree, using
``aetos_multiam.utils.mount_child`` for the actual USD-referencing + welding.
The underlying drone's own ``RobotBase``/``MultirotorBase`` machinery
(spawn/initialize/apply_action, rotor thrust) is reused unchanged from
OmniDrones -- this class only adds the arm/gripper layer on top.
"""

import logging
from typing import List, Optional, Sequence

import torch

from aetos_multiam.utils import mount_child
from .drones import get_drone_spec
from .manipulators import get_manipulator_spec
from .grippers import gripper_factory


class AerialManipulatorCfg:
    """Which drone/manipulator/gripper/controller to compose (robosuite ``robots=[...]``,
    ``controller_configs=...`` analog, collapsed into one dataclass since there's
    only one robot *type* in this benchmark)."""

    def __init__(
        self,
        drone_type: str,
        manipulator_type: Optional[str] = None,
        gripper_type: Optional[str] = None,
        controller_type: str = "default",
    ):
        self.drone_type = drone_type
        self.manipulator_type = manipulator_type
        self.gripper_type = gripper_type
        self.controller_type = controller_type


class AerialManipulatorBase:
    def __init__(self, cfg: AerialManipulatorCfg, name: Optional[str] = None):
        self.cfg = cfg
        self.drone_spec = get_drone_spec(cfg.drone_type)
        self.manipulator_spec = (
            get_manipulator_spec(cfg.manipulator_type) if cfg.manipulator_type else None
        )
        gripper_name = cfg.gripper_type or (
            self.manipulator_spec.default_gripper if self.manipulator_spec else None
        )
        self.gripper_spec = gripper_factory(gripper_name)

        self.drone_spec.check_asset()
        if self.manipulator_spec is not None:
            self.manipulator_spec.check_asset()
        if self.gripper_spec is not None and self.gripper_spec.usd_path is not None:
            self.gripper_spec.check_asset()

        self.name = name or f"am_{cfg.drone_type}"
        self.drone = self.drone_spec.drone_class(name=self.name)

        self.arm_joint_indices: Optional[torch.Tensor] = None
        self.gripper_joint_indices: Optional[torch.Tensor] = None

    def spawn(self, translations=None, orientations=None, prim_paths: Sequence[str] = None) -> List:
        if translations is None:
            translations = [(0.0, 0.0, 0.5)]
        prims = self.drone.spawn(translations, orientations, prim_paths)

        for prim in prims:
            drone_prim_path = prim.GetPath().pathString
            if self.manipulator_spec is None:
                continue
            if self.manipulator_spec.bundled_into_drone:
                logging.info(
                    "AerialManipulatorBase: manipulator '%s' is pre-baked into "
                    "the drone's own USD, nothing to mount.",
                    self.manipulator_spec.name,
                )
                continue

            manipulator_prim_path = mount_child(
                drone_prim_path,
                self.drone_spec.mount_body_name,
                self.manipulator_spec.usd_path,
                "manipulator",
                self.manipulator_spec.root_body_name,
                translation=self.manipulator_spec.mount_translation,
                orientation=self.manipulator_spec.mount_orientation,
            )

            if self.gripper_spec is not None and self.gripper_spec.usd_path is not None:
                mount_child(
                    manipulator_prim_path,
                    self.manipulator_spec.eef_body_name,
                    self.gripper_spec.usd_path,
                    "gripper",
                    self.gripper_spec.root_body_name,
                    translation=self.gripper_spec.mount_translation,
                    orientation=self.gripper_spec.mount_orientation,
                    joint_type=self.gripper_spec.mount_joint_type,
                    joint_axis=self.gripper_spec.mount_joint_axis,
                    joint_lower_limit=self.gripper_spec.mount_joint_lower_limit,
                    joint_upper_limit=self.gripper_spec.mount_joint_upper_limit,
                )
            elif self.gripper_spec is not None:
                logging.info(
                    "AerialManipulatorBase: gripper '%s' is bundled into '%s', nothing to mount.",
                    self.gripper_spec.name, self.manipulator_spec.name,
                )
        return prims

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        self.drone.initialize(prim_paths_expr=prim_paths_expr, track_contact_forces=track_contact_forces)
        self._resolve_joint_indices()

    def _resolve_joint_indices(self):
        """Look up arm/gripper DOF indices by name within the drone's merged articulation.

        Assumes ``mount_child`` folded the manipulator/gripper joints into the
        same ``ArticulationView`` the drone's own rotor joints live in (see the
        ArticulationRootAPI-stripping note in ``usd_compose.mount_child``).
        """
        dof_names = list(self.drone._view._dof_names)

        def _indices(names):
            missing = [n for n in names if n not in dof_names]
            if missing:
                raise RuntimeError(
                    f"Expected joints {missing} in the merged articulation but found "
                    f"only {dof_names}. Check the mounted USD's actual joint names."
                )
            return torch.tensor([dof_names.index(n) for n in names], device=self.drone.device)

        if self.manipulator_spec is not None and self.manipulator_spec.arm_joint_names:
            self.arm_joint_indices = _indices(self.manipulator_spec.arm_joint_names)
        if self.gripper_spec is not None and self.gripper_spec.joint_names:
            self.gripper_joint_indices = _indices(self.gripper_spec.joint_names)

    def apply_action(
        self,
        drone_action: torch.Tensor,
        arm_joint_targets: Optional[torch.Tensor] = None,
        gripper_joint_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write already-resolved per-part targets, each shaped ``(num_envs, num_joints)``
        (controller output, not raw policy actions -- see
        ``controllers.composite_controller.AerialManipulatorController``, which is
        what turns raw actions into ``arm_joint_targets``/``gripper_joint_targets``)."""
        throttle = self.drone.apply_action(drone_action)
        count = self.drone._view.count
        if arm_joint_targets is not None and self.arm_joint_indices is not None:
            self.drone._view.set_joint_position_targets(
                arm_joint_targets.reshape(count, -1), joint_indices=self.arm_joint_indices
            )
        if gripper_joint_targets is not None and self.gripper_joint_indices is not None:
            self.drone._view.set_joint_position_targets(
                gripper_joint_targets.reshape(count, -1), joint_indices=self.gripper_joint_indices
            )
        return throttle
