"""White bucket task prop -- PalatialSim/palatial-nv-asset-library asset
``D00436`` (hf dataset, revision ``69a87329``). Two rigid bodies
(``part_0``, the pail; ``part_1``, the swing handle) joined by one revolute
joint (axis X, authored limits -255..0 deg, position 0 = handle draped flat
against the rim).

Replaces the earlier ``bucket_utility_red_a02``-based ``Bucket`` object:
that asset's handle had an unreliable Hydra/Fabric render-sync bug (tensor
API position correct, rendered transform frozen) and, even when welded
statically to sidestep that, its bent-wire geometry never formed a classic
raised loop at any angle. ``D00436`` has neither problem -- the tensor API
and rendered transform tracked each other at every angle swept, and at
``-127.5 deg`` (the midpoint of its authored range) the handle forms a
clean symmetric raised arc above the rim, confirmed by screenshot.

The joint is still welded ``Fixed`` at spawn (not left live) for the same
reason as before: this prop only ever needs ONE fixed display pose, and a
0-DOF weld has no runtime DOF left for any future render-sync regression to
hit. Unlike the old bucket, the target pose isn't the joint's own authored
default -- ``weld_joint_at_angle`` first bakes the -127.5 deg rotation into
the joint's ``localRot1`` (pure quaternion math, no sim step needed) before
retyping to Fixed, reproducing exactly what live-commanding
``jointPosition = -127.5 deg`` renders (verified against the tensor-API
sweep this was derived from).
"""

from pathlib import Path

from aetos_multiam.utils import add_articulation_root, weld_joint_at_angle

from .object_base import AetosObject

_ASSETS = Path(__file__).resolve().parent / "assets"
USD_PATH = str(_ASSETS / "white_bucket" / "simready_usd" / "D00436.usd")

# Midpoint of the joint's authored -255..0 deg range -- forms a clean
# symmetric raised loop (see module docstring).
HANDLE_ANGLE_DEG = -127.5

# D00436's own mesh is close to the previous bucket_utility_red_a02 asset's
# native size already (both are ~0.2-0.3m real-world buckets) but not
# identical -- 1.038 matches D00436's body bbox to bucket_utility_red_a02's
# own bbox at ITS project scale of 0.7, via the geometric mean of the three
# per-axis ratios (1.021, 1.086, 1.008): measured directly from both assets'
# USD bounding boxes, not guessed. The extra *0.7 on top is a user-requested
# further shrink, not a size-matching correction.
SCALE = 1.038 * 0.7


class WhiteBucket(AetosObject):
    usd_path = USD_PATH
    # 0-DOF articulation after the weld: a body that's part of ANY
    # articulation, even a 0-DOF one, can no longer be matched by a plain
    # RigidPrimView ("did not match any rigid bodies") -- only
    # ArticulationView can register it. has_joints=True switches
    # AetosObject.initialize() to that.
    has_joints = True
    # Isaac Sim merges the referenced USD's own default prim (/World) into
    # the spawn root instead of nesting it one level down -- see
    # feedback-isaac-usd-default-prim-merge in project memory.
    handle_joint_rel_path = "textured_mesh_6a828810b7a9c4204ac539ac/Joints/joint_part_00__part_01"
    articulation_root_rel_path = "textured_mesh_6a828810b7a9c4204ac539ac"

    def __init__(self, name: str = "white_bucket", translation=(0.0, 0.0, 0.0), yaw_deg: float = 0.0):
        super().__init__(name, translation=translation, yaw_deg=yaw_deg, scale=SCALE)

    def spawn(self, template_env_ns: str) -> str:
        prim_path = super().spawn(template_env_ns)
        weld_joint_at_angle(f"{prim_path}/{self.handle_joint_rel_path}", HANDLE_ANGLE_DEG)
        add_articulation_root(f"{prim_path}/{self.articulation_root_rel_path}")
        return prim_path

    def initialize(self, prim_paths_expr: str) -> None:
        # ArticulationView must point at the prim carrying ArticulationRootAPI.
        super().initialize(f"{prim_paths_expr}/{self.articulation_root_rel_path}")
