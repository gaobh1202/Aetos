"""AetosEnv -- robosuite-shaped hooks layered onto OmniDrones's ``IsaacEnv``.

``IsaacEnv`` (torchrl's ``EnvBase`` + Isaac Sim's ``GridCloner``/
``SimulationContext`` bootstrap) is reused unchanged -- it already does what
robosuite's own ``MujocoEnv.__init__`` pipeline does (build model -> init
sim -> setup references -> setup observables), just for PhysX instead of
MuJoCo. This class only renames/re-splits that pipeline into the hook names
asked for ("通过 environment 模块注册环境类"), matching the exact call order
this repo's own ``hover.py`` already uses:

1. ``IsaacEnv.__init__`` calls ``_design_scene()`` (prims not live yet,
   nothing has been reset) -> here, ``_load_model()``: spawn arena + robots +
   objects, return global prim paths for collision filtering.
2. ``IsaacEnv.__init__`` resets the sim, then calls ``_set_specs()`` (shape
   info only -- the drone's ``state_spec``/``action_spec`` are available
   right after construction, no live view needed yet) -> here,
   ``_setup_observables()``: build observation/action/reward/agent specs.
3. Once ``IsaacEnv.__init__`` returns, ``AetosEnv.__init__`` calls
   ``_setup_references()``: grab live ``ArticulationView``/``RigidPrimView``
   handles (``drone.initialize()`` etc.) -- same point ``Hover.__init__``
   calls ``self.drone.initialize()``.

Per-step task logic is ``reward()`` and ``_check_success()``, called from
this class's ``_compute_reward_and_done()`` implementation of IsaacEnv's
abstract hook.
"""

import abc
from typing import List, Optional

import torch
from tensordict import TensorDict

from omni_drones.envs.isaac_env import IsaacEnv


class AetosEnv(IsaacEnv):
    #: robosuite ``REGISTERED_ENVS`` analog, re-exported as
    #: ``aetos_multiam.environments.REGISTERED_ENVS``. Implemented via
    #: ``__init_subclass__`` rather than a real metaclass (like robosuite's
    #: ``EnvMeta``) to avoid clashing with torchrl ``EnvBase``'s own metaclass.
    #: ``torchrl``'s ``EnvBase`` doesn't use ``abc.ABCMeta``, so
    #: ``__abstractmethods__`` isn't reliably populated here -- an explicit
    #: name blocklist (robosuite's own ``EnvMeta`` does the same for
    #: ``MujocoEnv``/``RobotEnv``/etc.) is used instead of introspection.
    REGISTERED_ENVS = {}
    _ABSTRACT_BASE_NAMES = {"AetosEnv", "AerialManipulationEnv", "CollaborativeAerialEnv"}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in AetosEnv._ABSTRACT_BASE_NAMES and not cls.__name__.startswith("_"):
            AetosEnv.REGISTERED_ENVS[cls.__name__] = cls

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self._setup_references()

    def _design_scene(self) -> Optional[List[str]]:
        return self._load_model()

    def _set_specs(self) -> None:
        self._setup_observables()

    @abc.abstractmethod
    def _load_model(self) -> Optional[List[str]]:
        """Spawn the arena, robot(s), and task objects into the template env.

        Returns the prim paths (e.g. the ground plane) that need global
        collision filtering across env clones -- robosuite's ``Task``
        composition step, re-expressed for USD prims instead of MJCF merge.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_observables(self) -> None:
        """Declare ``observation_spec``/``action_spec``/``reward_spec``/``agent_spec``."""
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_references(self) -> None:
        """Call ``.initialize()`` on every spawned robot/object to grab live views."""
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self) -> torch.Tensor:
        """Per-env scalar reward for the current step, shape ``(num_envs, 1)``
        (gets one more trailing dim added in ``_compute_reward_and_done`` to
        match ``reward_spec``'s ``(num_envs, 1, 1)``, same as ``hover.py``'s
        own ``reward.unsqueeze(-1)`` right before building its return dict)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _check_success(self) -> torch.Tensor:
        """Per-env success flag, shape ``(num_envs, 1)``."""
        raise NotImplementedError

    def _compute_reward_and_done(self) -> TensorDict:
        reward = self.reward()
        success = self._check_success()
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = success
        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
