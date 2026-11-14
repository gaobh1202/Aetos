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
from omni.isaac.core.simulation_context import SimulationContext

from aetos_multiam.utils import (
    add_articulation_root,
    apply_drone_livery,
    deactivate_prim,
    mount_child,
    normalize_joint_drives,
    set_joint_local_pose,
    strip_articulation_root,
)
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
        arm_dynamics_residual: float = 0.1,
        arm_locked_joints: Sequence[str] = (),
        enable_self_collisions: Optional[bool] = None,
        joint_pose_overrides: Optional[dict] = None,
        arm_write: Optional[str] = None,
    ):
        self.drone_type = drone_type
        self.manipulator_type = manipulator_type
        self.gripper_type = gripper_type
        self.controller_type = controller_type
        # How much of the arm's reaction on the drone to keep, and which joints
        # an operator may not drive. Both are properties of the ROBOT, measured
        # per preset (docs/ARM_DISTURBANCE.md) -- a scene's own
        # task.arm_dynamics_residual / task.arm_locked_joints override these,
        # and the CLI overrides both.
        self.arm_dynamics_residual = arm_dynamics_residual
        self.arm_locked_joints = tuple(arm_locked_joints)
        # None leaves whatever the asset authored. Needed because a vendored
        # asset may ship it off: hexa_scorpion.usd authors False, and its arm
        # then sweeps straight through the airframe.
        self.enable_self_collisions = enable_self_collisions
        # {joint prim path relative to the spawn root: (position, (w,x,y,z))},
        # re-authored at spawn. For re-mounting an arm that is pre-baked into a
        # vendored drone USD, where there is no mount_child call to adjust.
        self.joint_pose_overrides = dict(joint_pose_overrides or {})
        # "drive" | "teleport" | None (leave it to the caller). Set it only
        # where one of the two is measurably broken on this robot's asset --
        # see docs/TELEOP_CONTROL_MODES.md for both failure modes.
        if arm_write not in (None, "drive", "teleport"):
            raise ValueError(f"arm_write must be 'drive', 'teleport' or None, got {arm_write!r}")
        self.arm_write = arm_write


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
            self._pin_articulation_root_to_drone(drone_prim_path)
            if self.manipulator_spec is None:
                continue
            if self.manipulator_spec.bundled_into_drone:
                for body_name in self.manipulator_spec.deactivate_body_names:
                    deactivate_prim(f"{drone_prim_path}/{body_name}")

                if self.gripper_spec is not None and self.gripper_spec.usd_path is not None:
                    # Arm is already part of the drone's own USD -- the
                    # gripper mounts directly onto one of the drone's own
                    # bodies instead of onto a separately-mounted
                    # manipulator's eef (the deactivate_body_names loop
                    # above already cleared out whatever stock gripper used
                    # to occupy that body/DOF).
                    mount_child(
                        drone_prim_path,
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
                else:
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

        # A second pass, deliberately: the fixups traverse the whole robot, and
        # a mount_child'd arm does not exist until the loop above has run. Doing
        # this inside that loop silently skipped every separately-mounted arm --
        # which is exactly the one case the drive fixup was added for.
        for prim in prims:
            self._apply_asset_fixups(prim.GetPath().pathString)
            if self.drone_spec.livery:
                apply_drone_livery(prim.GetPath().pathString)
        return prims

    def _apply_asset_fixups(self, drone_prim_path: str) -> None:
        """Per-asset corrections that have to land before the first reset.

        All three are defects in vendored assets, not in this project, and all
        three present as "the robot looks wrong" rather than as an error.
        """
        import omni_drones.utils.kit as kit_utils

        # A drive authored under the wrong schema instance is invisible to the
        # solver -- dynamixel_2r's R1/R2 ship PhysicsDriveAPI:X, so hummingbird's
        # arm could not be commanded at all.
        normalize_joint_drives(drone_prim_path)

        for relative_path, (position, orientation) in self.cfg.joint_pose_overrides.items():
            set_joint_local_pose(f"{drone_prim_path}/{relative_path}", position, orientation)
            logging.info(
                "AerialManipulatorBase: re-mounted '%s' at pos=%s quat=%s",
                relative_path, tuple(position), tuple(orientation),
            )

        if self.manipulator_spec is not None and self.manipulator_spec.no_collide_with_drone:
            from omni.isaac.core.utils.stage import get_current_stage
            from pxr import UsdPhysics

            stage = get_current_stage()
            for body in self.manipulator_spec.no_collide_with_drone:
                prim = stage.GetPrimAtPath(f"{drone_prim_path}/manipulator/{body}")
                if not prim.IsValid():
                    raise RuntimeError(f"no_collide_with_drone: prim not found: {prim.GetPath()}")
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(
                    f"{drone_prim_path}/base_link")

        if self.manipulator_spec is not None and tuple(self.manipulator_spec.drive_gain_scale) != (1.0, 1.0):
            from omni.isaac.core.utils.stage import get_current_stage
            from pxr import Usd, UsdPhysics

            ks, ds = self.manipulator_spec.drive_gain_scale
            todo = set(self.manipulator_spec.arm_joint_names)
            for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(drone_prim_path)):
                drive = UsdPhysics.DriveAPI(prim, "angular")
                if prim.GetName() in todo and drive.GetStiffnessAttr():
                    drive.GetStiffnessAttr().Set(drive.GetStiffnessAttr().Get() * ks)
                    drive.GetDampingAttr().Set(drive.GetDampingAttr().Get() * ds)
                    todo.discard(prim.GetName())
            if todo:
                raise RuntimeError(f"drive_gain_scale: no angular drive found on {sorted(todo)}")

        if self.cfg.enable_self_collisions is not None:
            kit_utils.set_articulation_properties(
                self._articulation_prim_path(drone_prim_path),
                enable_self_collisions=self.cfg.enable_self_collisions,
            )

    def _articulation_prim_path(self, drone_prim_path: str) -> str:
        """Where the ArticulationRootAPI ended up -- base_link for every asset
        here once ``_pin_articulation_root_to_drone`` has run, but checked
        rather than assumed."""
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdPhysics

        candidate = f"{drone_prim_path}/base_link"
        prim = prim_utils.get_prim_at_path(candidate)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return candidate
        return drone_prim_path

    def _pin_articulation_root_to_drone(self, drone_prim_path: str) -> None:
        """Make the drone's mount body the articulation's floating base.

        Most drone USDs here author ``ArticulationRootAPI`` on ``base_link``
        itself, but some (hummingbird) author it on the USD's default prim,
        which Isaac merges onto the spawn root. That is harmless for a bare
        drone -- PhysX picks ``base_link`` anyway, it being the only body the
        rotor joints hang off -- but once ``mount_child`` welds an arm branch
        under the same ancestor, PhysX is free to pick an arm link as the
        floating base instead. Measured on hummingbird340_2dof_gripper: it
        picks ``first_segment``, and the drone tumbles (187 deg of yaw, 21 m
        of drift) with the arm held perfectly still. No-op for the presets
        that already carry the API on the mount body.
        """
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdPhysics

        import omni_drones.utils.kit as kit_utils

        # Deliberately not drone_spec.mount_body_name -- that is where the
        # *manipulator* attaches, which for amov950_airbot_bottom is link5,
        # an arm link. Every drone USD in this repo names its body base_link.
        body_path = f"{drone_prim_path}/base_link"
        body_prim = prim_utils.get_prim_at_path(body_path)
        root_prim = prim_utils.get_prim_at_path(drone_prim_path)
        if not body_prim.IsValid() or not root_prim.IsValid():
            return
        if body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return
        if not root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return
        strip_articulation_root(drone_prim_path)
        add_articulation_root(body_path)
        # RobotBase.spawn already authored the solver settings on the old
        # prim, where they are now orphaned -- re-apply on the new root.
        props = self.drone.articulation_props
        kit_utils.set_articulation_properties(
            body_path,
            enable_self_collisions=props.enable_self_collisions,
            solver_position_iteration_count=props.solver_position_iteration_count,
            solver_velocity_iteration_count=props.solver_velocity_iteration_count,
        )
        logging.info(
            "AerialManipulatorBase: moved ArticulationRootAPI from '%s' to '%s' "
            "so the drone body, not an arm link, is the floating base.",
            drone_prim_path, body_path,
        )

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False,
                   reset_rest_pose: bool = True):
        """Grab live views for this robot.

        ``reset_rest_pose=False`` skips the second ``SimulationContext.reset()``
        described further down -- which is mandatory when SEVERAL robots share
        one scene, because that reset invalidates every *other* robot's view
        handles too, not just this one's. The caller then owns the sequence:
        ``initialize(reset_rest_pose=False)`` on every robot, one
        ``SimulationContext.instance().reset()``, then ``reattach()`` on every
        robot. ``examples/show_am_action.py --preset all`` does exactly that.
        """
        self._prim_paths_expr = prim_paths_expr
        self._track_contact_forces = track_contact_forces
        self.drone.initialize(prim_paths_expr=prim_paths_expr, track_contact_forces=track_contact_forces)

        # params["mass"] comes from the *bare* drone's yaml, but the arm and
        # gripper mounted at spawn time are part of the vehicle the controller
        # has to lift, and LeePositionController reads this straight into its
        # gravity feedforward. hummingbird340_2dof_gripper: 0.716 declared vs
        # 0.858 real, a 16.5% deficit, measured as a 0.49m vertical sag with
        # the arm held perfectly still. Total mass is what matters here, and
        # freeze_arm_dynamics preserves it, so the order of the two is free.
        body_names = list(self.drone._view._body_names)
        if body_names[0] != "base_link":
            raise RuntimeError(
                f"{self.__class__.__name__}: PhysX made '{body_names[0]}' the "
                f"articulation's floating base instead of 'base_link' -- the drone "
                f"will tumble with the arm held still. Body order: {body_names}. "
                "See _pin_articulation_root_to_drone."
            )

        total_mass = float(self.drone._view.get_body_masses().sum(-1).reshape(-1)[0])
        if abs(total_mass - float(self.drone.params["mass"])) > 1e-6:
            logging.info(
                "AerialManipulatorBase: controller mass %.4f -> %.4f kg (the "
                "mounted arm/gripper are not in the drone's own yaml).",
                float(self.drone.params["mass"]), total_mass,
            )
            self.drone.params["mass"] = total_mass

        if self.manipulator_spec is not None and reset_rest_pose:
            # The *first* sim.reset() after mount_child() welds a new branch
            # (arm [+ gripper]) onto the drone's existing articulation --
            # PhysX's initial rest-pose solve for that merged articulation is
            # wrong by exactly the welded branch's own internal joint offset
            # (confirmed empirically: an arm with a -0.368m fixed-joint
            # offset from its mount point snaps the ENTIRE rigid group,
            # drone included, by +0.368m/some axis rotation on the very
            # first physics step, then holds perfectly still forever after
            # -- a one-time mis-solve, not a real dynamic instability).
            # A SECOND reset, after the articulation view already exists,
            # re-solves the same rest pose correctly and it then never
            # drifts again.
            SimulationContext.instance().reset()
            # That second reset tears down and recreates the simulation
            # context's physics simulation view -- the drone's own
            # ArticulationView object (self.drone._view) survives as a
            # Python object, but its cached _physics_view handle (fetched
            # during the FIRST initialize() above) is now stale, bound to a
            # simulation view that no longer exists. USD-level pose reads
            # (XformCache) don't go through that handle, so this looked
            # harmless under every check this project ran before -- but any
            # PhysX tensor-API call (get_dof_velocities/
            # get_dof_position_targets, hit by apply_action's
            # set_joint_velocities/set_joint_position_targets) fails with
            # "Failed to get ... from backend" against the stale handle.
            # Re-running initialize() recreates the ArticulationView fresh,
            # bound to the current (post-reset) simulation view.
            #
            # And that is precisely why a multi-robot scene has to opt out: the
            # teardown is global, so robot N's reset leaves robots 0..N-1
            # holding stale handles. Reading one then segfaults inside
            # get_masses -- it does not even raise. See reset_rest_pose above.
            self.drone.initialize(
                prim_paths_expr=prim_paths_expr, track_contact_forces=track_contact_forces
            )

        self._resolve_joint_indices()

    def reattach(self) -> None:
        """Re-create this robot's views after someone reset the
        SimulationContext. Pairs with ``initialize(reset_rest_pose=False)``."""
        self.drone.initialize(
            prim_paths_expr=self._prim_paths_expr,
            track_contact_forces=self._track_contact_forces,
        )
        self._resolve_joint_indices()

    @property
    def num_controlled_arm_joints(self) -> int:
        """How many DOFs the arm controller will actually drive.

        Tasks size their action/observation specs in ``_set_specs()``, which
        runs BEFORE ``initialize()`` has populated ``arm_joint_indices`` -- so
        this has to be answerable from static spec data alone, AND it has to
        agree with what ``_resolve_joint_indices`` ends up producing. When the
        two disagree, ``AerialManipulatorController.split_action`` (which
        derives the drone's slice as ``action_dim - arm - gripper``) cuts the
        action vector at the wrong offset: AerialButtonGrasp was handing
        ``AmovDrone`` a 3-wide command for a 4-rotor drone that way, because
        amov_drone_arm's two baked-in gripper DOFs were undeclared.
        ``_resolve_joint_indices`` asserts the two agree.
        """
        if self.manipulator_spec is not None and self.manipulator_spec.arm_joint_names:
            return len(self.manipulator_spec.arm_joint_names)
        # No ManipulatorSpec -- the arm is baked into the drone's own USD, and
        # _resolve_joint_indices falls back to the drone class's own
        # auto-detected "every non-rotor DOF", which includes the baked-in
        # gripper's finger joints. Both lists count.
        return (
            len(self.drone_spec.bundled_arm_joint_names)
            + len(self.drone_spec.bundled_gripper_joint_names)
        )

    @property
    def num_gripper_action_dims(self) -> int:
        """1 if a separate gripper controller consumes a trailing scalar from
        the action vector, else 0. A gripper baked into the drone's own USD has
        no ``GripperSpec`` -- its finger joints are counted in
        ``num_controlled_arm_joints`` instead, not here."""
        return 1 if self.gripper_spec is not None else 0

    @property
    def teleop_joint_indices(self) -> Optional[torch.Tensor]:
        """Every operator-drivable DOF as one flat index tensor: arm joints
        first, gripper joints last.

        That ordering is the contract every teleop device maps onto -- the
        SO-ARM leader's ``JOINT_ORDER``, the keyboard's joint picker, and an
        exoskeleton/AVP retargeter all emit arm-then-gripper. ``None`` when
        the robot has no arm at all.
        """
        if self.arm_joint_indices is None:
            return None
        if self.gripper_joint_indices is None:
            return self.arm_joint_indices
        return torch.cat([self.arm_joint_indices, self.gripper_joint_indices])

    @property
    def teleop_joint_names(self) -> List[str]:
        """DOF names matching ``teleop_joint_indices``, for operator-facing UI."""
        indices = self.teleop_joint_indices
        if indices is None:
            return []
        dof_names = list(self.drone._view._dof_names)
        return [dof_names[i] for i in indices.tolist()]

    def hold_joints(self, targets: torch.Tensor, zero_velocity: bool = False) -> None:
        """Drive every teleop DOF to ``targets`` through its PD drive."""
        indices = self.teleop_joint_indices
        if indices is None:
            return
        count = self.drone._view.count
        targets = targets.reshape(count, 1, -1)
        self.drone._view.set_joint_position_targets(targets, joint_indices=indices)
        if zero_velocity:
            self.drone._view.set_joint_velocities(
                torch.zeros_like(targets), joint_indices=indices
            )

    def teleport_joints(self, positions: torch.Tensor) -> None:
        """Write every teleop DOF's position state directly, bypassing its drive.

        The lag-free path a 1:1 leader mapping needs, and the only path that
        works at all on so_arm100's ``shoulder_pan`` (see
        ``docs/ARM_DISTURBANCE.md``). Costs a visible flicker and kicks the
        drone, which is what ``arm_dynamics_residual`` exists to damp.
        Deliberately writes no joint velocity -- doing so flips the drone, see
        the measured table in ``examples/aerial_manipulator_control_demo.py``.
        """
        indices = self.teleop_joint_indices
        if indices is None:
            return
        count = self.drone._view.count
        self.drone._view.set_joint_positions(
            positions.reshape(count, 1, -1), joint_indices=indices
        )

    #: Cached home pose for ``reset_arm_joints``; class-level so it exists
    #: before ``initialize`` has run.
    _arm_home = None

    def reset_arm_joints(self, env_ids=None) -> None:
        """Put every teleop DOF back to its home pose: zero, clamped into each
        joint's own range.

        Belongs here, not on the drone: ``AmovDrone`` happens to carry a method
        of the same name because its arm is baked into its own USD, but every
        other preset mounts its arm at runtime and the drone class knows
        nothing about it -- calling the drone's version crashes with
        ``'_AmbenchQuadScorpion' object has no attribute 'reset_arm_joints'``.
        Clamped rather than plain zeros because a joint whose range excludes
        zero (airbot's ``joint3`` is [-0.094, 3.161]) would otherwise start
        outside its own limit.
        """
        indices = self.teleop_joint_indices
        if indices is None:
            return
        if self._arm_home is None:
            limits = self.drone._view.get_dof_limits()[0, 0][indices].to(self.drone.device)
            self._arm_home = torch.zeros(len(indices), device=self.drone.device).clamp(
                limits[:, 0], limits[:, 1]
            )
        count = self.drone._view.count
        home = self._arm_home.expand(count, 1, -1).contiguous()
        self.drone._view.set_joint_positions(home, joint_indices=indices)
        self.drone._view.set_joint_velocities(torch.zeros_like(home), joint_indices=indices)
        self.drone._view.set_joint_position_targets(home, joint_indices=indices)

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
        elif getattr(self.drone, "arm_joint_indices", None) is not None:
            # No separate ManipulatorSpec -- arm is baked into the drone's
            # own USD (e.g. amov_drone_arm) and the drone class itself
            # already auto-detected these DOFs (every non-rotor joint, see
            # AmovDrone.initialize()). Mirror that here so apply_action()
            # and every task built on this wrapper see one consistent
            # arm_joint_indices regardless of which path the arm came from.
            self.arm_joint_indices = self.drone.arm_joint_indices
        if self.gripper_spec is not None and self.gripper_spec.joint_names:
            self.gripper_joint_indices = _indices(self.gripper_spec.joint_names)

        resolved = 0 if self.arm_joint_indices is None else len(self.arm_joint_indices)
        if resolved != self.num_controlled_arm_joints:
            raise RuntimeError(
                f"{self.name}: arm_joint_indices resolved to {resolved} DOFs, but "
                f"num_controlled_arm_joints -- what every task sizes its action and "
                f"observation specs from, before initialize() runs -- says "
                f"{self.num_controlled_arm_joints}. Declare the missing joints on the "
                f"DroneSpec/ManipulatorSpec; left unfixed this silently splits the "
                f"action vector at the wrong offset instead of failing here."
            )

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
