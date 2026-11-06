"""ShelfBucketGrasp -- same shelf scene as ``ShelfButtonGrasp``, but the
gearwheel/button/power-drill cluster on the shelf's top tier is replaced by
one white bucket prop (see ``objects/white_bucket.py``), and the tote box is
dropped entirely. Defaults to the ``iris650_soarm100`` preset instead of
``amov950_airbot_bottom``.

By default the drone body itself isn't flown -- it's welded to the world at
spawn time (``pin_body_to_world`` on its ``base_link``, same mechanism
``examples/amteleop_soarm_leader.py`` uses to hold a robot still for arm-only
teleop) at the same ``ROBOT_POS``/``ROBOT_YAW_DEG`` ``ShelfButtonGrasp`` uses
for its own (flown) robot. The action space only covers the arm (+ gripper, if
the preset has one) as a result -- a constant zero rotor command is spliced in
internally before handing the full vector to ``AerialManipulatorController``
(which still expects one).

``task.weld_robot=false`` drops the weld and puts the rotor commands back at
the FRONT of the action vector, so the same scene can be flown -- that's what
``examples/aerial_manipulator_control_demo.py --task ShelfBucketGrasp`` uses to
teleop this scene's own declared robot. Nothing else about the task changes;
the reward/success logic never looked at the drone.

The source ``ShelfButtonGrasp`` scene is a static layout only (no
reward/success logic of its own) -- the eef-to-bucket distance reward/
success below is the same placeholder "reach" stand-in
``ShelfButtonGrasp`` uses, just retargeted at the bucket instead of the
button.
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
from aetos_multiam.objects import WhiteBucket
from aetos_multiam.robots import AerialManipulatorBase
from aetos_multiam.scenes import ShelfArena
from aetos_multiam.utils import pin_body_to_world, yaw_deg_to_quat

from ..aerial_manipulation_env import AerialManipulationEnv


class ShelfBucketGrasp(AerialManipulationEnv):
    r"""Move a stationary aerial manipulator's arm to reach a shelf-mounted bucket.

    ## Observation
    - `rpos_eef_bucket` (3): eef position relative to the bucket.
    - `drone_state` (depends on the drone model): see `MultirotorBase.get_state()`.
    - `arm_joint_pos` (num_arm_joints): current arm joint angles.
    - `time_encoding` (optional, 4): episode progress encoding.

    ## Reward
    - `reward_distance`: shaped on `rpos_eef_bucket`'s norm.
    - `reward_success`: one-off bonus the step the eef gets within `success_distance`.

    ## Episode end
    Success: eef within `success_distance` of the bucket (TODO: replace with
    real grasp detection once gripper-contact sensing lands). Otherwise ends
    at `max_episode_length`.
    """

    # Same shelf, same template-env-relative robot pose as ShelfButtonGrasp
    # (see that class for how ROBOT_POS/ROBOT_YAW_DEG were read back from the
    # source scene) -- the drone just doesn't need to fly to reach it here.
    # BUCKET_POS centers the bucket across the x-span the gearwheel/button/
    # power-drill cluster it replaces used to occupy (-0.007 to 0.377), with
    # a bit of z clearance above that cluster's own 1.35-1.44 baseline so it
    # drops onto the shelf surface instead of spawning embedded in it.
    BUCKET_POS = (0.185, 0.045351, 1.40)
    ROBOT_POS = (-0.099038, 0.438106, 2.002114)
    ROBOT_YAW_DEG = -90.0  # face -y, toward the shelf's open front

    def __init__(self, cfg, headless):
        self.reward_distance_scale = float(cfg.task.get("reward_distance_scale", 1.2))
        self.reward_success_weight = float(cfg.task.get("reward_success_weight", 10.0))
        self.success_distance = float(cfg.task.get("success_distance", 0.05))
        self.time_encoding = bool(cfg.task.get("time_encoding", True))
        self.weld_robot = bool(cfg.task.get("weld_robot", True))

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
            prims = am.spawn(
                translations=[self._default_spawn_translation(i)],
                orientations=[yaw_deg_to_quat(self.ROBOT_YAW_DEG)],
            )
            # Weld in place (see module docstring) -- must happen before the
            # caller's first sim.reset(), same timing as every other
            # structural USD edit in this project.
            if self.weld_robot:
                pin_body_to_world(f"{prims[0].GetPath().pathString}/base_link")
            self.aerial_manipulators.append(am)
        self.drone = self.aerial_manipulators[0]

        global_paths += self._load_objects()
        return global_paths

    def _load_objects(self) -> List[str]:
        self.bucket = WhiteBucket(name="white_bucket", translation=self.BUCKET_POS)
        self.bucket.spawn(self.template_env_ns)
        return []

    def _setup_object_references(self) -> None:
        self.bucket.initialize(f"{self.env_ns}/env_.*/white_bucket")

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
        # No rotor dims while the drone is pinned to the world (see module
        # docstring) -- there's nothing for a policy to usefully command on the
        # drone side then.
        action_dim = (0 if self.weld_robot else num_rotors) + num_arm_joints
        action_dim += self.drone.num_gripper_action_dims

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
        if self.weld_robot:
            # The drone is pinned to the world (see module docstring) -- the
            # policy's action only covers arm(+gripper); splice in a constant
            # zero rotor command so AerialManipulatorController still sees the
            # full drone+arm(+gripper) vector it expects to split.
            num_rotors = self.drone.drone.num_rotors
            zero_rotor = torch.zeros(*action.shape[:-1], num_rotors, device=self.device)
            action = torch.cat([zero_rotor, action], dim=-1)
        self.controller.run(self.drone, action)

    def _compute_state_and_obs(self) -> TensorDict:
        eef_pos, _ = self.eef_view.get_world_poses(clone=True)
        eef_pos = eef_pos.reshape(self.num_envs, 1, 3) - self.envs_positions.unsqueeze(1)

        bucket_pos, _ = self.bucket.get_world_pose(clone=True)
        bucket_pos = bucket_pos.reshape(self.num_envs, 1, 3) - self.envs_positions.unsqueeze(1)

        self.rpos_eef_bucket = bucket_pos - eef_pos

        drone_state = self.drone.drone.get_state()
        arm_pos = (
            self.drone.drone._view.get_joint_positions()[..., self.drone.arm_joint_indices]
            if self.drone.arm_joint_indices is not None
            else torch.zeros(self.num_envs, 1, 0, device=self.device)
        )
        if arm_pos.dim() == 2:
            arm_pos = arm_pos.unsqueeze(1)

        obs = [self.rpos_eef_bucket, drone_state, arm_pos]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return TensorDict({"agents": {"observation": obs}}, self.batch_size)

    def reward(self) -> torch.Tensor:
        distance = torch.norm(self.rpos_eef_bucket, dim=-1)
        reward_distance = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        reward_success = self.reward_success_weight * self._check_success().float()
        reward = reward_distance + reward_success
        return torch.nan_to_num(reward, nan=0.0)

    def _check_success(self) -> torch.Tensor:
        distance = torch.norm(self.rpos_eef_bucket, dim=-1)
        return (distance <= self.success_distance).reshape(self.num_envs, 1)
