"""AetosTaskSpec -- declaration schema for future multi-AM demo generation.

Modeled directly on NVIDIA mimicgen's own ``mimicgen.configs.task_spec.MG_TaskSpec``
(found locally at ``/home/benhua/DexSim/mimicgen/mimicgen/configs/task_spec.py``),
**not** dexmimicgen's own bespoke replay scripts (``demo_generation/*.py``) --
the Explore pass into both found the real ``MG_TaskSpec`` + ``env_interfaces``
pattern (ordered subtasks, each ending on a named termination *signal*) to be
the reusable, generalizable design, while dexmimicgen's own generation code
is hand-tuned geometric heuristics specific to its dexterous-hand tasks.

Nothing in ``aetos_multiam`` consumes this yet -- there is no generator.
This is the schema a future ``ButtonGrasp``-like task (or a multi-AM
collaboration task) would declare its subtask sequence with, once automated
generation is actually built. See the real ``MG_TaskSpec``/``env_interfaces``
pair for the full generation-time contract (``get_object_poses()`` /
``get_subtask_term_signals()`` on an env interface class, one 0->1 edge per
subtask boundary).
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AetosSubtask:
    object_ref: str
    """Which object/AM this subtask is defined relative to (env-specific key)."""
    subtask_term_signal: Optional[str]
    """Name of a boolean signal the env reports; its 0->1 edge ends this subtask."""
    subtask_term_offset_range: Tuple[int, int] = (0, 0)
    selection_strategy: str = "random"
    selection_strategy_kwargs: Optional[Dict[str, Any]] = None
    action_noise: float = 0.0
    num_interpolation_steps: int = 5
    num_fixed_steps: int = 0
    apply_noise_during_interpolation: bool = False


class AetosTaskSpec:
    """Ordered list of ``AetosSubtask``, one per agent-visible phase of a task."""

    def __init__(self):
        self.subtasks: List[AetosSubtask] = []

    def add_subtask(
        self,
        object_ref: str,
        subtask_term_signal: Optional[str],
        subtask_term_offset_range: Tuple[int, int] = (0, 0),
        selection_strategy: str = "random",
        selection_strategy_kwargs: Optional[Dict[str, Any]] = None,
        action_noise: float = 0.0,
        num_interpolation_steps: int = 5,
        num_fixed_steps: int = 0,
        apply_noise_during_interpolation: bool = False,
    ) -> None:
        self.subtasks.append(
            AetosSubtask(
                object_ref=object_ref,
                subtask_term_signal=subtask_term_signal,
                subtask_term_offset_range=subtask_term_offset_range,
                selection_strategy=selection_strategy,
                selection_strategy_kwargs=selection_strategy_kwargs,
                action_noise=action_noise,
                num_interpolation_steps=num_interpolation_steps,
                num_fixed_steps=num_fixed_steps,
                apply_noise_during_interpolation=apply_noise_during_interpolation,
            )
        )

    def serialize(self) -> str:
        return json.dumps(
            {f"subtask_{i + 1}": asdict(s) for i, s in enumerate(self.subtasks)}, indent=2
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AetosTaskSpec":
        spec = cls()
        data = json.loads(json_str)
        for key in sorted(data, key=lambda k: int(k.split("_")[-1])):
            entry = dict(data[key])
            entry["subtask_term_offset_range"] = tuple(entry.get("subtask_term_offset_range", (0, 0)))
            spec.subtasks.append(AetosSubtask(**entry))
        return spec

    def __len__(self) -> int:
        return len(self.subtasks)

    def __getitem__(self, idx: int) -> AetosSubtask:
        return self.subtasks[idx]

    def __iter__(self):
        return iter(self.subtasks)
