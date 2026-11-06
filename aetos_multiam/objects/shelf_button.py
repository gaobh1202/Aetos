"""ShelfButton task prop -- migrated from raids-am-simulator's shelf_grasp
scene. Distinct from ``objects.button.Button`` (AM-bench's push_button.usd,
with a real ``PlungerSlideJoint``) -- this is a plain two-part rigid button
model (``model_redbutton.usd``), no plunger joint, used as the
``ShelfButtonGrasp`` task's grasp target. All three rigid sub-bodies bake a
non-standard xform that needs normalizing before PhysX will simulate them --
see ``aetos_multiam.utils.usd_fixups``.

E_body_16 is actually an articulation ROOT: the two sub-buttons
(E_button_01_13/E_button_02_6) hang off it via PrismaticJoints. For
whole-object pose replay (teleporting the main body every frame), the
whole prop gets flipped kinematic -- ``aetos_multiam.utils.make_kinematic``
already deactivates any internal joint it finds (PhysX refuses to create
a joint whose both ends are about to be kinematic), and
``strip_articulation_root`` below removes E_body_16's ArticulationRootAPI
(PhysX just warns-and-ignores it on a kinematic body, but the tensor API
then fails to register the body at all without the schema actually
removed). The two sub-buttons end up frozen wherever they spawn -- nothing
drives them once their joint is gone -- acceptable: they're decorative
compared to the main body.
"""

from pathlib import Path

from aetos_multiam.utils import normalize_rigid_root_xform, strip_articulation_root

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "button" / "model_redbutton.usd")


class ShelfButton(AetosObject):
    usd_path = USD_PATH
    root_body_names = ("E_body_16", "E_button_01_13", "E_button_02_6")

    def __init__(self, name: str = "shelf_button", translation=(0.0, 0.0, 0.0)):
        super().__init__(name, translation=translation)

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        for body_name in self.root_body_names:
            normalize_rigid_root_xform(f"{prim_path}/{body_name}")
        strip_articulation_root(f"{prim_path}/{self.root_body_names[0]}")
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        # The RigidBodyAPI prim is nested one level under the reference
        # wrapper -- a RigidPrimView on the wrapper itself matches zero
        # rigid bodies. Track the main body (E_body_16); the two sub-button
        # parts are secondary.
        super().initialize(f"{prim_paths_expr}/{self.root_body_names[0]}")
