"""ShelfButtonGrasp -- migrated from raids-am-simulator's shelf_grasp scene
(``raids_am_Scene/build_isaacsim_scene.py``): a 4-tier chrome shelf holding
a gearwheel, a button, a power drill, and a tote box, with an aerial
manipulator parked in front of it.

Defaults to the ``amov950_airbot_bottom`` preset (``amov_airbot_gripper``'s
drone+arm combo) -- swap it via ``cfg.task.aerial_manipulators[0].preset``
(or a hydra override, e.g. ``aerial_manipulators.0.preset=<other_preset>``),
no code change needed, same as every other ``AerialManipulationEnv`` task.

The source scene is a static layout only (no reward/success logic of its
own) -- the eef-to-button distance reward/success below is a placeholder
"reach" task standing in for real grasp detection (gripper contact), marked
here as the extension point for whichever pick task comes next.
"""

from typing import List

import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.utils.torchrl import AgentSpec
from omni_drones.views import RigidPrimView

from aetos_multiam.controllers import (
    AerialManipulatorController,
    BinaryGripperController,
    JointPositionController,
)
from aetos_multiam.objects import Gearwheel, PowerDrill, ShelfButton, ToteBox
from aetos_multiam.robots import AerialManipulatorBase
from aetos_multiam.scenes import ShelfArena
from aetos_multiam.utils import yaw_deg_to_quat

from ..aerial_manipulation_env import AerialManipulationEnv


class ShelfButtonGrasp(AerialManipulationEnv):
    r"""Fly an aerial manipulator up to a shelf-mounted button and reach it.

    ## Observation
    - `rpos_eef_button` (3): eef position relative to the button.
    - `drone_state` (depends on the drone model): see `MultirotorBase.get_state()`.
    - `arm_joint_pos` (num_arm_joints): current arm joint angles.
    - `time_encoding` (optional, 4): episode progress encoding.

    ## Reward
    - `reward_distance`: shaped on `rpos_eef_button`'s norm.
    - `reward_success`: one-off bonus the step the eef gets within `success_distance`.

    ## Episode end
    Success: eef within `success_distance` of the button (TODO: replace with
    real grasp detection once gripper-contact sensing lands). Otherwise ends
    at `max_episode_length`.
    """

    # Hand-tuned layout read back from the source scene's shelf_grasp.usd
    # (see build_isaacsim_scene.py) -- shelf sits at the template env origin.
    GEARWHEEL_POS = (-0.006728, 0.038057, 1.386416)
    BUTTON_POS = (0.172601, 0.045351, 1.365782)
    DRILL_POS = (0.376657, 0.021583, 1.441908)
    DRILL_YAW_DEG = 180.0
    TOTE_BOX_POS = (-0.276033, 0.009358, 1.345964)
    TOTE_BOX_YAW_DEG = 90.0
    ROBOT_POS = (-0.099038, 0.438106, 2.002114)
    ROBOT_YAW_DEG = -90.0  # face -y, toward the shelf's open front

    def __init__(self, cfg, headless):
        self.reward_distance_scale = float(cfg.task.get("reward_distance_scale", 1.2))
        self.reward_success_weight = float(cfg.task.get("reward_success_weight", 10.0))
        self.success_distance = float(cfg.task.get("success_distance", 0.05))
        self.time_encoding = bool(cfg.task.get("time_encoding", True))

        super().__init__(cfg, headless)

    # -- scene -----------------------------------------------------------

    def _build_arena(self) -> ShelfArena:
        return ShelfArena()

    def _default_spawn_translation(self, index: int):
        return self.ROBOT_POS

    def _load_model(self) -> List[str]:
        # Override to set the robot's yaw (AerialManipulationEnv._load_model
        # only lets _default_spawn_translation vary per agent, not orientation).
        self.arena = self._build_arena()
        global_paths = list(self.arena.spawn(self.template_env_ns))

        self.aerial_manipulators = []
        for i, am_cfg in enumerate(self.am_cfgs):
            am = AerialManipulatorBase(am_cfg, name=f"am_{i}")
            am.spawn(
                translations=[self._default_spawn_translation(i)],
                orientations=[yaw_deg_to_quat(self.ROBOT_YAW_DEG)],
            )
            self.aerial_manipulators.append(am)
        self.drone = self.aerial_manipulators[0]

        global_paths += self._load_objects()
        return global_paths

    def _load_objects(self) -> List[str]:
        self.gearwheel = Gearwheel(name="gearwheel", translation=self.GEARWHEEL_POS)
        self.gearwheel.spawn(self.template_env_ns)
        self.button = ShelfButton(name="shelf_button", translation=self.BUTTON_POS)
        self.button.spawn(self.template_env_ns)
        self.power_drill = PowerDrill(
            name="power_drill", translation=self.DRILL_POS, yaw_deg=self.DRILL_YAW_DEG
        )
        self.power_drill.spawn(self.template_env_ns)
        self.tote_box = ToteBox(
            name="tote_box", translation=self.TOTE_BOX_POS, yaw_deg=self.TOTE_BOX_YAW_DEG
        )
        self.tote_box.spawn(self.template_env_ns)
        return []

    def _setup_object_references(self) -> None:
        self.gearwheel.initialize(f"{self.env_ns}/env_.*/gearwheel")
        self.button.initialize(f"{self.env_ns}/env_.*/shelf_button")
        self.power_drill.initialize(f"{self.env_ns}/env_.*/power_drill")
        self.tote_box.initialize(f"{self.env_ns}/env_.*/tote_box")

    def _setup_references(self) -> None:
        super()._setup_references()

        manipulator_spec = self.drone.manipulator_spec
        gripper_spec = self.drone.gripper_spec
        # get_dof_limits() is (count, group, num_dof, 2) -- squeeze both
        # leading dims, not just the first, or arm_joint_indices (into the
        # num_dof axis) silently indexes the wrong axis and goes out of
        # bounds (CUDA device-side assert under the GPU pipeline).
        dof_limits = self.drone.drone._view.get_dof_limits()[0, 0]  # (num_dof, 2), same across envs

        arm_controller = None
        if self.drone.arm_joint_indices is not None:
            limits = dof_limits[self.drone.arm_joint_indices]
            arm_controller = JointPositionController(
                joint_lower=limits[:, 0].tolist(),
                joint_upper=limits[:, 1].tolist(),
                device=self.device,
            )
        gripper_controller = BinaryGripperController(gripper_spec) if gripper_spec is not None else None

        self.controller = AerialManipulatorController(
            arm_controller=arm_controller,
            gripper_controller=gripper_controller,
        )

        if manipulator_spec is not None:
            # mount_child() references the manipulator as a SIBLING of the
            # drone's own top-level prim (not nested under mount_body_name)
            # -- see usd_compose.mount_child's own docstring.
            eef_prim_glob = (
                f"{self.env_ns}/env_.*/{self.drone.name}_0/manipulator/{manipulator_spec.eef_body_name}"
            )
        else:
            # Arm baked into the drone's own USD (no ManipulatorSpec) --
            # bodies are flat siblings directly under the drone's own prim,
            # not nested under mount_body_name/manipulator/...
            eef_prim_glob = f"{self.env_ns}/env_.*/{self.drone.name}_0/{self.drone.drone_spec.eef_body_name}"
        self.eef_view = RigidPrimView(eef_prim_glob, reset_xform_properties=False)
        self.eef_view.initialize()

    # -- specs -------------------------------------------------------------

    def _setup_observables(self) -> None:
        num_rotors = self.drone.drone.num_rotors
        drone_state_dim = 19 + num_rotors
        num_arm_joints = self.drone.num_controlled_arm_joints

        self.time_encoding_dim = 4 if self.time_encoding else 0
        observation_dim = 3 + drone_state_dim + num_arm_joints + self.time_encoding_dim
        action_dim = num_rotors + num_arm_joints + self.drone.num_gripper_action_dims

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
            }),
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": BoundedTensorSpec(-1, 1, (1, action_dim), device=self.device),
            }),
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1), device=self.device),
            }),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["aerial_manipulator"] = AgentSpec(
            "aerial_manipulator", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

    # -- step / reset --------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        self.drone.drone._reset_idx(env_ids, train=True)
        if self.drone.arm_joint_indices is not None:
            # AmovDrone has a built-in reset_arm_joints() for its own baked-in
            # arm; a mounted ManipulatorSpec (e.g. so_arm100 on iris_650) has
            # no such method on the underlying drone class -- fall back to a
            # plain zero-pose reset via the tensor API directly.
            reset_arm_joints = getattr(self.drone.drone, "reset_arm_joints", None)
            if reset_arm_joints is not None:
                reset_arm_joints(env_ids)
            else:
                n = len(env_ids)
                zeros = torch.zeros(n, len(self.drone.arm_joint_indices), device=self.device)
                view = self.drone.drone._view
                view.set_joint_positions(zeros, env_indices=env_ids, joint_indices=self.drone.arm_joint_indices)
                view.set_joint_velocities(torch.zeros_like(zeros), env_indices=env_ids, joint_indices=self.drone.arm_joint_indices)
                view.set_joint_position_targets(zeros, env_indices=env_ids, joint_indices=self.drone.arm_joint_indices)

    def _pre_sim_step(self, tensordict: TensorDict) -> None:
        action = tensordict[("agents", "action")]
        self.controller.run(self.drone, action)

    def _compute_state_and_obs(self) -> TensorDict:
        eef_pos, _ = self.eef_view.get_world_poses(clone=True)
        eef_pos = eef_pos.reshape(self.num_envs, 1, 3) - self.envs_positions.unsqueeze(1)

        button_pos, _ = self.button.get_world_pose(clone=True)
        button_pos = button_pos.reshape(self.num_envs, 1, 3) - self.envs_positions.unsqueeze(1)

        self.rpos_eef_button = button_pos - eef_pos

        drone_state = self.drone.drone.get_state()
        arm_pos = (
            self.drone.drone._view.get_joint_positions()[..., self.drone.arm_joint_indices]
            if self.drone.arm_joint_indices is not None
            else torch.zeros(self.num_envs, 1, 0, device=self.device)
        )
        if arm_pos.dim() == 2:
            arm_pos = arm_pos.unsqueeze(1)

        obs = [self.rpos_eef_button, drone_state, arm_pos]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return TensorDict({"agents": {"observation": obs}}, self.batch_size)

    def reward(self) -> torch.Tensor:
        distance = torch.norm(self.rpos_eef_button, dim=-1)
        reward_distance = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        reward_success = self.reward_success_weight * self._check_success().float()
        reward = reward_distance + reward_success
        return torch.nan_to_num(reward, nan=0.0)

    def _check_success(self) -> torch.Tensor:
        distance = torch.norm(self.rpos_eef_button, dim=-1)
        return (distance <= self.success_distance).reshape(self.num_envs, 1)
