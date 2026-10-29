"""Button task object -- real asset, reused directly from AM-bench.

``push_button.usd`` is plain USD (geometry + a PhysX slide joint), so it's
simulator-agnostic and safe to reuse outside IsaacLab. Body/joint names and
the pressed-threshold value (``0.004`` m) come straight from AM-bench's own
``press_button_env.py``/``press_button_env_cfg.py``, which read this exact
file.
"""

import torch

from .object_base import AetosObject

USD_PATH = "/home/benhua/ambench/source/ambench/ambench/assets/objects/push_button.usd"


class Button(AetosObject):
    usd_path = USD_PATH
    has_joints = True

    joint_name = "PlungerSlideJoint"
    plunger_body_name = "plunger"
    pressed_threshold = 0.004  # meters, from AM-bench's press_button_env_cfg.py

    def initialize(self, prim_paths_expr: str) -> None:
        super().initialize(prim_paths_expr)
        dof_names = list(self._view._dof_names)
        self.joint_index = dof_names.index(self.joint_name)

    def get_plunger_joint_position(self) -> torch.Tensor:
        return self._view.get_joint_positions()[:, self.joint_index]

    def is_pressed(self) -> torch.Tensor:
        return self.get_plunger_joint_position() >= self.pressed_threshold
