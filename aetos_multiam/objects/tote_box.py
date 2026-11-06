"""ToteBox task prop -- migrated from raids-am-simulator's shelf_grasp
scene. Its rigid root bakes a non-standard xform that needs normalizing
before PhysX will simulate it -- see ``aetos_multiam.utils.usd_fixups``.
"""

from pathlib import Path

from aetos_multiam.utils import normalize_rigid_root_xform

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "tote_box" / "usd_src" / "model_ed1240a5d47249d7_0.usd")


class ToteBox(AetosObject):
    usd_path = USD_PATH
    root_body_name = "E_tote_box_1"

    def __init__(self, name: str = "tote_box", translation=(0.0, 0.0, 0.0), yaw_deg: float = 90.0):
        super().__init__(name, translation=translation, yaw_deg=yaw_deg, scale=0.7)

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        normalize_rigid_root_xform(f"{prim_path}/{self.root_body_name}")
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        # The RigidBodyAPI prim is nested one level under the reference
        # wrapper -- a RigidPrimView on the wrapper itself matches zero
        # rigid bodies.
        super().initialize(f"{prim_paths_expr}/{self.root_body_name}")
