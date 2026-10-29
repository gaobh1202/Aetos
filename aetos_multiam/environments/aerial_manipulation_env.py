"""AerialManipulationEnv -- spawns N aerial manipulators + arena + objects.

robosuite ``RobotEnv``/``ManipulationEnv`` analog: owns the generic "how many
robots, what are they, where do they spawn, which arena" plumbing shared by
every task in this benchmark -- concrete tasks only add objects, reward, and
success logic. Agents are declared entirely in config
(``cfg.task.aerial_manipulators``, a list with one entry per agent), so a
heterogeneous multi-AM team (e.g. a hexacopter + a quadcopter) is just a
longer list, not a code change -- see ``cfg/task/Aerial*.yaml`` and
``robots/presets.py`` for the named combos those lists reference.
"""

import abc
from typing import List, Optional, Sequence

from aetos_multiam.robots import AerialManipulatorBase, AerialManipulatorCfg, get_preset
from aetos_multiam.scenes import Arena
from .base_env import AetosEnv


def _resolve_agent_cfg(agent_cfg) -> AerialManipulatorCfg:
    """One entry of ``cfg.task.aerial_manipulators`` -> an ``AerialManipulatorCfg``.

    Either ``{preset: <name>}`` (looked up in ``robots.PRESETS``) or the raw
    fields directly (``drone_type``/``manipulator_type``/``gripper_type``/
    ``controller_type``) -- the preset form is what every ``cfg/task/Aerial*.yaml``
    in this repo actually uses, the raw form exists for one-off experiments
    that don't warrant a named preset.
    """
    preset_name = agent_cfg.get("preset", None)
    if preset_name is not None:
        return get_preset(preset_name)
    return AerialManipulatorCfg(
        drone_type=agent_cfg.drone_type,
        manipulator_type=agent_cfg.get("manipulator_type", None),
        gripper_type=agent_cfg.get("gripper_type", None),
        controller_type=agent_cfg.get("controller_type", "default"),
    )


class AerialManipulationEnv(AetosEnv):
    def __init__(self, cfg, headless):
        self.am_cfgs: List[AerialManipulatorCfg] = [
            _resolve_agent_cfg(agent_cfg) for agent_cfg in cfg.task.aerial_manipulators
        ]
        self.num_agents = len(self.am_cfgs)
        super().__init__(cfg, headless)

    def _load_model(self) -> Optional[List[str]]:
        self.arena = self._build_arena()
        global_paths = list(self.arena.spawn(self.template_env_ns))

        self.aerial_manipulators: List[AerialManipulatorBase] = []
        for i, am_cfg in enumerate(self.am_cfgs):
            am = AerialManipulatorBase(am_cfg, name=f"am_{i}")
            am.spawn(translations=[self._default_spawn_translation(i)])
            self.aerial_manipulators.append(am)
        self.drone = self.aerial_manipulators[0]  # convenience alias for single-agent tasks

        global_paths += self._load_objects()
        return global_paths

    def _build_arena(self) -> Arena:
        return Arena()

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        """Spread agents out along Y by default; override for task-specific layouts."""
        return (0.0, 1.5 * index, 1.0)

    @abc.abstractmethod
    def _load_objects(self) -> List[str]:
        """Spawn task-specific objects into the template env; return global prim paths."""
        raise NotImplementedError

    def _setup_references(self) -> None:
        for am in self.aerial_manipulators:
            am.initialize()
        self._setup_object_references()

    @abc.abstractmethod
    def _setup_object_references(self) -> None:
        """Call ``.initialize()`` on every spawned task object."""
        raise NotImplementedError
