"""ButtonGrasp -- the end-to-end proof task: one AmovDrone950 + airbot_play
flies to a wall-mounted button and presses it.

Uses AM-bench's own ``press_button`` task concept and its ``push_button.usd``
asset directly (see ``aetos_multiam.objects.button.Button``), reimplemented
on raw ``omni.isaac.core`` instead of IsaacLab. This is the one task this
first pass actually runs end-to-end; every other drone/manipulator/gripper
combination in the registries is scaffolding only (see docs/ARCHITECTURE.md).
"""

import logging
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
from aetos_multiam.objects import Button
from aetos_multiam.scenes import WallArena

from ..aerial_manipulation_env import AerialManipulationEnv


class ButtonGrasp(AerialManipulationEnv):
    r"""Fly an aerial manipulator up to a wall-mounted button and press it.

    ## Observation
    - `rpos_eef_button` (3): eef position relative to the button's plunger.
    - `drone_state` (depends on the drone model): see the underlying `MultirotorBase.get_state()`.
    - `arm_joint_pos` (num_arm_joints): current arm joint angles.
    - `button_plunger_pos` (1): how far the button has already been pressed.
    - `time_encoding` (optional, 4): episode progress encoding.

    ## Reward
    - `reward_distance`: shaped on `rpos_eef_button`'s norm.
    - `reward_press`: bonus proportional to the plunger's own joint position.
    - `reward_success`: one-off bonus the step the button crosses `pressed_threshold`.

    ## Episode end
    Success: the button is pressed (`Button.is_pressed()`). Otherwise ends at `max_episode_length`.
    """

    def __init__(self, cfg, headless):
        self.reward_distance_scale = float(cfg.task.get("reward_distance_scale", 1.2))
        self.reward_press_weight = float(cfg.task.get("reward_press_weight", 1.0))
        self.reward_success_weight = float(cfg.task.get("reward_success_weight", 10.0))
        self.time_encoding = bool(cfg.task.get("time_encoding", True))

        wall_cfg = cfg.task.get("wall", {})
        self._wall_translation = tuple(wall_cfg.get("translation", (2.0, 0.0, 1.0)))
        self._wall_size = tuple(wall_cfg.get("size", (0.1, 2.0, 2.0)))

        super().__init__(cfg, headless)

    # -- scene -----------------------------------------------------------

    def _build_arena(self) -> WallArena:
        return WallArena(wall_translation=self._wall_translation, wall_size=self._wall_size)

    def _default_spawn_translation(self, index: int):
        wall_x, wall_y, wall_z = self._wall_translation
        return (wall_x - 1.5, wall_y, wall_z)

    def _load_objects(self) -> List[str]:
        wall_x, wall_y, wall_z = self._wall_translation
        wall_half_thickness = self._wall_size[0] / 2.0
        button_translation = (wall_x - wall_half_thickness - 0.02, wall_y, wall_z)
        self.button = Button(name="button", translation=button_translation)
        self.button.spawn(self.template_env_ns)
        return []

    def _setup_object_references(self) -> None:
        self.button.initialize(f"{self.env_ns}/env_.*/button")

    def _setup_references(self) -> None:
        super()._setup_references()

        manipulator_spec = self.drone.manipulator_spec
        gripper_spec = self.drone.gripper_spec
        dof_limits = self.drone.drone._view.get_dof_limits()[0]  # (num_dof, 2), same across envs

        arm_controller = None
        if manipulator_spec is not None and self.drone.arm_joint_indices is not None:
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

        # Track the manipulator's eef body world pose for the reward/obs.
        eef_prim_glob = (
            f"{self.env_ns}/env_.*/{self.drone.name}_0/"
            f"{self.drone.drone_spec.mount_body_name}/manipulator/{manipulator_spec.eef_body_name}"
        )
        self.eef_view = RigidPrimView(eef_prim_glob, reset_xform_properties=False)
        self.eef_view.initialize()

    # -- specs -------------------------------------------------------------

    def _setup_observables(self) -> None:
        # NOTE: by this point _load_model() has already run (see base_env.py's
        # docstring on IsaacEnv's call order) so self.drone and its specs
        # exist, even though .initialize() (live views) hasn't happened yet --
        # MultirotorBase.num_rotors and ManipulatorSpec.arm_joint_names are
        # both set at construction time, no live view needed.
        num_rotors = self.drone.drone.num_rotors
        drone_state_dim = 19 + num_rotors
        num_arm_joints = (
            len(self.drone.manipulator_spec.arm_joint_names) if self.drone.manipulator_spec else 0
        )

        self.time_encoding_dim = 4 if self.time_encoding else 0
        observation_dim = 3 + drone_state_dim + num_arm_joints + 1 + self.time_encoding_dim
        action_dim = num_rotors + num_arm_joints + 1

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
            self.drone.drone.reset_arm_joints(env_ids)

    def _pre_sim_step(self, tensordict: TensorDict) -> None:
        # Keep the ("agents", "action") tensor's (num_envs, 1, action_dim) shape
        # as-is -- the trailing "1" matches drone.shape and is what
        # MultirotorBase.apply_action's `actions.expand(*self.shape, ...)` call
        # expects; AerialManipulatorBase.apply_action reshapes the arm/gripper
        # slices itself before writing joint targets.
        action = tensordict[("agents", "action")]
        self.controller.run(self.drone, action)

    def _compute_state_and_obs(self) -> TensorDict:
        eef_pos, _ = self.eef_view.get_world_poses(clone=True)
        eef_pos = eef_pos.reshape(self.num_envs, 1, 3) - self.envs_positions.unsqueeze(1)

        # Coarse proxy: the button's own root pose, not its (moving) plunger
        # body specifically -- close enough for distance-shaping purposes.
        button_pos, _ = self.button.get_world_pose(clone=True)
        button_pos = button_pos.reshape(self.num_envs, 1, 3) - self.envs_positions.unsqueeze(1)
        self.button_plunger_pos = self.button.get_plunger_joint_position().reshape(self.num_envs, 1, 1)

        self.rpos_eef_button = button_pos - eef_pos

        drone_state = self.drone.drone.get_state()
        arm_pos = (
            self.drone.drone._view.get_joint_positions()[..., self.drone.arm_joint_indices]
            if self.drone.arm_joint_indices is not None
            else torch.zeros(self.num_envs, 1, 0, device=self.device)
        )
        if arm_pos.dim() == 2:
            arm_pos = arm_pos.unsqueeze(1)

        obs = [self.rpos_eef_button, drone_state, arm_pos, self.button_plunger_pos]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return TensorDict({"agents": {"observation": obs}}, self.batch_size)

    def reward(self) -> torch.Tensor:
        distance = torch.norm(self.rpos_eef_button, dim=-1)
        reward_distance = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        reward_press = self.reward_press_weight * self.button_plunger_pos.squeeze(-1)
        reward_success = self.reward_success_weight * self._check_success().float()
        reward = reward_distance + reward_press + reward_success
        return torch.nan_to_num(reward, nan=0.0)

    def _check_success(self) -> torch.Tensor:
        return self.button.is_pressed().reshape(self.num_envs, 1)
