"""Shared scaffolding for multi-AM collaboration tasks.

Agents here are genuinely heterogeneous -- different drone/arm/gripper per
agent, declared entirely in ``cfg.task.aerial_manipulators`` (see
``cfg/task/Aerial*.yaml`` and ``robots/presets.py``). That's a deliberate
departure from this repo's own ``Formation``/``Rearrange`` envs, which spawn
N *identical* drones as one batched articulation (one Python object,
``shape=(num_envs, n_agents)``); a shared-width stacked ``("agents",
"observation")`` tensor doesn't work once agents can have different arm
joint counts or even different drone state dims. So each agent gets its own
top-level key (``f"agent_{i}"``) in the observation/action/reward specs
instead of one shared "agents" dimension.

None of the five tasks built on this base have a real task-object asset yet
(a handover target, assembly fixture, shared payload, or cabinet) --
``_load_objects``/``_setup_object_references`` are no-ops here. Reward and
success are a generic placeholder (hold position near a per-agent goal) so
the config-driven heterogeneous spawning is at least exercisable end-to-end
for the one preset that can actually spawn today
(``amov950_airbot_bottom``, used by ``AerialTransport`` and
``AerialAssemblyTriple``) -- swap in real task logic once the mixed-team
presets have USD assets and/or real task objects exist, following
``ButtonGrasp``'s pattern.
"""

from typing import List, Sequence

import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.utils.torchrl import AgentSpec

from ..aerial_manipulation_env import AerialManipulationEnv


class CollaborativeAerialEnv(AerialManipulationEnv):
    def __init__(self, cfg, headless):
        self.goal_radius = float(cfg.task.get("goal_radius", 0.5))
        self.time_encoding = bool(cfg.task.get("time_encoding", True))
        super().__init__(cfg, headless)

    # -- scene: no task object yet, see module docstring ---------------------

    def _load_objects(self) -> List[str]:
        return []

    def _setup_object_references(self) -> None:
        pass

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        return (0.0, 1.5 * index, 1.0)

    def _goal_position(self, index: int) -> Sequence[float]:
        """Per-agent goal; straight above its spawn point by default. Override
        per task for layouts that actually mean something (e.g. a shared
        midpoint for a handover)."""
        x, y, z = self._default_spawn_translation(index)
        return (x, y, z + 1.0)

    # -- specs ------------------------------------------------------------

    def _setup_observables(self) -> None:
        self.time_encoding_dim = 4 if self.time_encoding else 0
        self._agent_keys = [f"agent_{i}" for i in range(self.num_agents)]

        agents_obs = {}
        agents_action = {}
        agents_reward = {}
        for key, am in zip(self._agent_keys, self.aerial_manipulators):
            num_rotors = am.drone.num_rotors
            drone_state_dim = 19 + num_rotors
            num_arm_joints = len(am.manipulator_spec.arm_joint_names) if am.manipulator_spec else 0
            action_dim = num_rotors + num_arm_joints + (1 if am.gripper_spec is not None else 0)
            obs_dim = 3 + drone_state_dim + num_arm_joints + self.time_encoding_dim

            agents_obs[key] = CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, obs_dim), device=self.device),
            })
            agents_action[key] = CompositeSpec({
                "action": BoundedTensorSpec(-1, 1, (1, action_dim), device=self.device),
            })
            agents_reward[key] = CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1), device=self.device),
            })

        self.observation_spec = CompositeSpec(agents_obs).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec(agents_action).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec(agents_reward).expand(self.num_envs).to(self.device)

        for key in self._agent_keys:
            self.agent_spec[key] = AgentSpec(
                key, 1,
                observation_key=(key, "observation"),
                action_key=(key, "action"),
                reward_key=(key, "reward"),
            )

    # -- step / reset --------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        for am in self.aerial_manipulators:
            am.drone._reset_idx(env_ids, train=True)
            if am.arm_joint_indices is not None:
                am.drone.reset_arm_joints(env_ids)

    def _pre_sim_step(self, tensordict: TensorDict) -> None:
        for key, am in zip(self._agent_keys, self.aerial_manipulators):
            # Keep the (num_envs, 1, action_dim) shape as-is -- the trailing
            # "1" matches drone.shape and is what MultirotorBase.apply_action's
            # `actions.expand(*self.shape, ...)` call expects (see the same
            # note in environments/single/button_grasp.py's _pre_sim_step).
            action = tensordict[(key, "action")]
            am.apply_action(action[..., : am.drone.num_rotors])
            # Arm/gripper targets intentionally not driven here yet -- no
            # per-agent controller is wired up in this generic base (see
            # module docstring); concrete tasks that need the arm moving
            # should build one the way ButtonGrasp does.

    def _compute_state_and_obs(self) -> TensorDict:
        self._rpos_to_goal = []
        obs_dict = {}
        for i, (key, am) in enumerate(zip(self._agent_keys, self.aerial_manipulators)):
            drone_state = am.drone.get_state()
            goal = torch.as_tensor(self._goal_position(i), device=self.device).reshape(1, 1, 3)
            rpos = goal - drone_state[..., :3]
            self._rpos_to_goal.append(rpos)

            arm_pos = (
                am.drone._view.get_joint_positions()[..., am.arm_joint_indices]
                if am.arm_joint_indices is not None
                else torch.zeros(self.num_envs, 1, 0, device=self.device)
            )
            if arm_pos.dim() == 2:
                arm_pos = arm_pos.unsqueeze(1)

            pieces = [rpos, drone_state, arm_pos]
            if self.time_encoding:
                t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
                pieces.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
            obs = torch.nan_to_num(torch.cat(pieces, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            obs_dict[key] = {"observation": obs}

        return TensorDict(obs_dict, self.batch_size)

    def reward(self) -> dict:
        """Per-agent reward, shape ``(num_envs, 1)`` each -- gets one more
        trailing dim in ``_compute_reward_and_done`` to match
        ``reward_spec``'s ``(num_envs, 1, 1)``, same convention as
        ``AetosEnv.reward()`` (see that class's docstring), just keyed by
        agent instead of a single tensor."""
        rewards = {}
        for key, rpos in zip(self._agent_keys, self._rpos_to_goal):
            distance = torch.norm(rpos, dim=-1)  # (num_envs, 1)
            rewards[key] = 1.0 / (1.0 + torch.square(distance))
        return rewards

    def _check_success(self) -> torch.Tensor:
        all_close = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        for rpos in self._rpos_to_goal:
            distance = torch.norm(rpos, dim=-1).reshape(self.num_envs, 1)
            all_close &= distance < self.goal_radius
        return all_close

    def _compute_reward_and_done(self) -> TensorDict:
        rewards = self.reward()
        success = self._check_success()
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = success
        reward_dict = {key: {"reward": r.unsqueeze(-1)} for key, r in rewards.items()}
        return TensorDict(
            {
                **reward_dict,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
