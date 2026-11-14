# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ArmTrajectory: drone fixed in place, arm follows a predefined sinusoidal trajectory.

Run (example):
    OMNIDRONES_AMOV_USD_PATH=/home/benhua/OmniDrones/amov_drone_arm_1.usd \
    python scripts/train.py task=ArmTrajectory algo=ppo headless=false
"""

import logging
import math

import omni.isaac.core.utils.prims as prim_utils
import torch

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec


class ArmTrajectory(IsaacEnv):
    """Drone body fixed at a configurable height; arm joints follow sinusoidal trajectories.

    This environment is intended for visual inspection of arm motion in Isaac Sim.
    There is no RL policy — the drone is pinned by resetting its root-link pose and
    velocity to the spawn pose every physics substep, while arm joint position targets
    are updated each control step according to per-joint sinusoidal parameters.

    Config keys (cfg.task):
        fixed_height (float): height at which the drone is pinned (default 1.5 m).
        arm_trajectory.joints (list): per-joint dicts with keys:
            amplitude (float, rad): peak displacement (default 0.3).
            frequency (float, Hz): oscillation frequency (default 0.3).
            phase    (float, rad): initial phase offset (default 0.0).
        arm_trajectory.stiffness (float): PD stiffness for arm joints (default 500).
        arm_trajectory.damping   (float): PD damping  for arm joints (default 50).
    """

    def __init__(self, cfg, headless):
        self._fixed_height = float(cfg.task.get("fixed_height", 1.5))

        # Parse per-joint trajectory parameters.
        traj_cfg = cfg.task.get("arm_trajectory", None)
        if traj_cfg is not None:
            joints_cfg = traj_cfg.get("joints", [])
            self._arm_stiffness = float(traj_cfg.get("stiffness", 500.0))
            self._arm_damping = float(traj_cfg.get("damping", 50.0))
        else:
            joints_cfg = []
            self._arm_stiffness = 500.0
            self._arm_damping = 50.0

        # Store as plain lists; converted to tensors after drone.initialize().
        self._joints_cfg = list(joints_cfg)

        super().__init__(cfg, headless)

        self.drone.initialize()

        # Build per-joint trajectory tensors now that we know num_arm_joints.
        n = self.drone.num_arm_joints
        amplitudes = torch.zeros(n, device=self.device)
        frequencies = torch.zeros(n, device=self.device)
        phases = torch.zeros(n, device=self.device)
        for i, jcfg in enumerate(self._joints_cfg):
            if i >= n:
                break
            amplitudes[i] = float(jcfg.get("amplitude", 0.3))
            frequencies[i] = float(jcfg.get("frequency", 0.3))
            phases[i] = float(jcfg.get("phase", 0.0))
        # Fill remaining joints with defaults if fewer configs than joints.
        for i in range(len(self._joints_cfg), n):
            amplitudes[i] = 0.3
            frequencies[i] = 0.3 + i * 0.05  # slight frequency spread for visual variety

        self._traj_amplitudes = amplitudes   # (num_arm_joints,)
        self._traj_frequencies = frequencies
        self._traj_phases = phases

        logging.info(
            "ArmTrajectory: %d arm joints | amplitudes=%s | frequencies=%s | phases=%s",
            n,
            amplitudes.tolist(),
            frequencies.tolist(),
            phases.tolist(),
        )

        # Apply PD gains to arm joints.
        if n > 0 and self.drone.arm_joint_indices is not None:
            stiffness = torch.full(
                (self.drone._view.count, n), self._arm_stiffness, device=self.device
            )
            damping = torch.full(
                (self.drone._view.count, n), self._arm_damping, device=self.device
            )
            self.drone._view.set_gains(
                stiffness, damping, joint_indices=self.drone.arm_joint_indices
            )
            logging.info(
                "ArmTrajectory: set arm PD gains stiffness=%.1f damping=%.1f",
                self._arm_stiffness,
                self._arm_damping,
            )

        # Cache the fixed world pose for each env (spawn position + fixed_height).
        # init_poses shape: (num_envs, 1, 3) pos, (num_envs, 1, 4) rot
        self._fixed_pos, self._fixed_rot = self.drone.get_world_poses(clone=True)
        self._fixed_vel = torch.zeros_like(self.drone.get_velocities())

        logging.info(
            "ArmTrajectory: drone pinned at height=%.2f m (world z of env_0: %.3f)",
            self._fixed_height,
            float(self._fixed_pos[0, 0, 2].item()),
        )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils

        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        self.drone.spawn(translations=[(0.0, 0.0, self._fixed_height)])
        return ["/World/defaultGroundPlane"]

    # ------------------------------------------------------------------
    # Specs (minimal — no RL policy)
    # ------------------------------------------------------------------

    def _set_specs(self):
        # Dummy single-float observation and action so the harness is satisfied.
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, 1), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((1, 1), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        # Pin drone at fixed position.
        self.drone.set_world_poses(
            self._fixed_pos[env_ids], self._fixed_rot[env_ids], env_ids
        )
        self.drone.set_velocities(self._fixed_vel[env_ids], env_ids)

        # Reset arm joints to zero.
        if self.drone.arm_joint_indices is not None:
            zeros = torch.zeros(
                len(env_ids), self.drone.num_arm_joints, device=self.device
            )
            self.drone._view.set_joint_positions(
                zeros, env_indices=env_ids, joint_indices=self.drone.arm_joint_indices
            )
            self.drone._view.set_joint_velocities(
                zeros, env_indices=env_ids, joint_indices=self.drone.arm_joint_indices
            )
            self.drone._view.set_joint_position_targets(
                zeros, env_indices=env_ids, joint_indices=self.drone.arm_joint_indices
            )

        self.stats[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def _pre_sim_step(self, tensordict: TensorDictBase):
        # Pin drone body: teleport root link back to fixed pose every substep.
        # This counteracts gravity and any residual forces without needing thrust.
        self.drone.set_world_poses(self._fixed_pos, self._fixed_rot)
        self.drone.set_velocities(self._fixed_vel)

        # Apply sinusoidal arm trajectory.
        self._apply_arm_trajectory()

    def _apply_arm_trajectory(self):
        if self.drone.arm_joint_indices is None or self.drone.num_arm_joints == 0:
            return

        # Time in seconds for each env (progress_buf counts control steps).
        t = self.progress_buf.float() * (self.dt * self.substeps)  # (num_envs,)

        # pos[i, j] = A_j * sin(2π * f_j * t_i + φ_j)
        # t: (num_envs,) → (num_envs, 1)
        # frequencies/amplitudes/phases: (num_arm_joints,) → (1, num_arm_joints)
        pos = (
            self._traj_amplitudes.unsqueeze(0)
            * torch.sin(
                2.0 * math.pi * self._traj_frequencies.unsqueeze(0) * t.unsqueeze(1)
                + self._traj_phases.unsqueeze(0)
            )
        )  # (num_envs, num_arm_joints)

        self.drone._view.set_joint_position_targets(
            pos, joint_indices=self.drone.arm_joint_indices
        )

    # ------------------------------------------------------------------
    # Observations / rewards (minimal)
    # ------------------------------------------------------------------

    def _compute_state_and_obs(self):
        dummy_obs = torch.zeros(self.num_envs, 1, 1, device=self.device)
        return TensorDict(
            {
                "agents": {"observation": dummy_obs},
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        reward = torch.zeros(self.num_envs, 1, 1, device=self.device)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["return"] += reward.squeeze(-1)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {"reward": reward},
                "done": truncated,
                "terminated": torch.zeros_like(truncated),
                "truncated": truncated,
            },
            self.batch_size,
        )
