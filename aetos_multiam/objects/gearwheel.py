"""Gearwheel task prop -- migrated from raids-am-simulator's shelf_grasp
scene (``raids_am_Scene/build_isaacsim_scene.py``). The source asset bakes a
non-standard xform on its rigid root, which needs normalizing before PhysX
will actually simulate it -- see ``aetos_multiam.utils.usd_fixups``.
"""

from pathlib import Path

from aetos_multiam.utils import normalize_rigid_root_xform

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "gearwheel" / "model_gearwhee3.usd")


class Gearwheel(AetosObject):
    usd_path = USD_PATH
    root_body_name = "E_body_24"

    def __init__(self, name: str = "gearwheel", translation=(0.0, 0.0, 0.0)):
        super().__init__(name, translation=translation, scale=2.0)

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        normalize_rigid_root_xform(f"{prim_path}/{self.root_body_name}")
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        # The RigidBodyAPI prim is nested one level under the reference
        # wrapper (see spawn()'s normalize_rigid_root_xform target) -- a
        # RigidPrimView on the wrapper itself matches zero rigid bodies.
        super().initialize(f"{prim_paths_expr}/{self.root_body_name}")
