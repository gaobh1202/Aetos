"""PowerDrill task prop -- migrated from raids-am-simulator's shelf_grasp
scene. Ships with a stray world-anchoring FixedJoint that must be
deactivated, otherwise it can't actually be picked up.
"""

from pathlib import Path

from aetos_multiam.utils import deactivate_prim

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "power_drill" / "power_drill.usd")


class PowerDrill(AetosObject):
    usd_path = USD_PATH
    root_body_name = "power_drill"  # nested one level under the reference wrapper
    stray_joint_rel_path = "power_drill/FixedJoint"

    def __init__(self, name: str = "power_drill", translation=(0.0, 0.0, 0.0), yaw_deg: float = 180.0):
        super().__init__(name, translation=translation, yaw_deg=yaw_deg)

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        deactivate_prim(f"{prim_path}/{self.stray_joint_rel_path}")
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        # The RigidBodyAPI prim is nested one level under the reference
        # wrapper (same nesting the stray FixedJoint's own path implies) --
        # a RigidPrimView on the wrapper itself matches zero rigid bodies.
        super().initialize(f"{prim_paths_expr}/{self.root_body_name}")
