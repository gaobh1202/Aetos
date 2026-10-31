"""Import dynamixel_2r_arm.xml (see split_combo.py) via Isaac Sim's own
omni.importer.mjcf -- plain rigid links + 2 hinge joints, no flexcomp/
equality (both already stripped out along with the gripper), so this is
exactly the case that importer handles cleanly (confirmed for the other 3
MJCF-only bundles in this project already).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "dynamixel_2r_arm.xml"
OUT_DIR = HERE.parent.parent / "meshes" / "usd" / "dynamixel_2r"
DEST = OUT_DIR / "dynamixel_2r_arm.usd"

from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from pxr import Usd

OUT_DIR.mkdir(parents=True, exist_ok=True)

ctx = omni.usd.get_context()
ctx.new_stage()

status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
print("ImportConfig methods:", [m for m in dir(import_config) if m.startswith("set_")], flush=True)
import_config.set_fix_base(True)
import_config.set_import_inertia_tensor(True)

result = omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path=str(MJCF_PATH),
    import_config=import_config,
    # Not "/dynamixel_2r_arm" -- the importer nests the MJCF's own root body
    # (named "arm_carrier") one level under whatever prim_path wrapper we
    # give it, so reusing that same name here would produce a confusing
    # double .../arm_carrier/arm_carrier. A distinct wrapper name keeps it
    # to one segment: .../DynamixelArm/arm_carrier.
    prim_path="/DynamixelArm",
    dest_path=str(DEST),
)
print("import result:", result, flush=True)

from pxr import UsdPhysics, Sdf

stage = Usd.Stage.Open(str(DEST))

# Vendoring cleanup (same pattern as this project's other manipulators,
# e.g. airbot_play's own root_joint->world chain): the importer's own
# set_fix_base(True) welded the root body to WORLD and put
# PhysicsArticulationRootAPI on the wrapper Xform one level above it --
# both exist only for standalone testing. mount_child() authors the real
# weld (drone base_link -> this arm's root) and its own articulation
# handling at spawn time, so both would be redundant/conflicting left in.
root_joint_path = "/DynamixelArm/arm_carrier/joints/rootJoint_arm_carrier"
root_joint = stage.GetPrimAtPath(root_joint_path)
if root_joint.IsValid():
    stage.RemovePrim(Sdf.Path(root_joint_path))
    print("removed stray world-anchor joint at", root_joint_path, flush=True)

wrapper = stage.GetPrimAtPath("/DynamixelArm/arm_carrier")
if wrapper.HasAPI(UsdPhysics.ArticulationRootAPI):
    wrapper.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print("removed ArticulationRootAPI from", wrapper.GetPath(), flush=True)

# The importer also authors a second, separate ArticulationRootAPI holder,
# an empty "worldBody" placeholder sibling to arm_carrier with no children
# of its own -- pure standalone-testing scaffolding (same category as the
# stray root joint above), not part of the arm's own kinematic tree at all.
world_body_path = "/DynamixelArm/worldBody"
world_body = stage.GetPrimAtPath(world_body_path)
if world_body.IsValid():
    stage.RemovePrim(Sdf.Path(world_body_path))
    print("removed empty worldBody placeholder at", world_body_path, flush=True)

# The importer applies DriveAPI to every joint but leaves stiffness/damping
# at 0 (inert) since the trimmed MJCF carries no <actuator> block (dropped
# along with the gripper's own actuator/equality block -- see
# split_combo.py). Set them from the ORIGINAL MJCF's own
# <position class="dynamixel_2r" kp="50" dampratio="1" .../> actuator:
# kp maps directly to USD stiffness; dampratio=1 (critical damping) has no
# simple closed form without replicating MuJoCo's own per-joint-inertia
# calculation, so damping=5.0 N*m*s/rad is a documented approximation, not
# a derived value -- tune at the controller level if tracking looks off.
for jname in ("R1", "R2"):
    joint = stage.GetPrimAtPath(f"/DynamixelArm/arm_carrier/joints/{jname}")
    joint.GetAttribute("drive:X:physics:stiffness").Set(50.0)
    joint.GetAttribute("drive:X:physics:damping").Set(5.0)
print("set R1/R2 drive stiffness=50, damping=5.0 (approximate)", flush=True)

# arm_carrier and tool_adapter are pure mount-point bodies in the source
# MJCF -- no <geom>/<inertial> of their own (arm_carrier's only geom,
# carrier_plate, is contype=0/conaffinity=0, visual-only, so there's no
# collider for the importer to derive mass from either). The importer
# authored physics:mass=0.0 explicitly for both, which PhysX flags at
# runtime as "possibly invalid inertia tensor... and a negative mass,
# small sphere approximated inertia was used" -- confirmed live: this is
# not just a warning, it destabilizes the whole merged articulation once
# physics actually steps (the combo tumbles and falls instead of holding
# its spawn pose). Fixed with a small, physically-negligible explicit mass
# so PhysX has something valid to work with.
for name in ("arm_carrier", "tool_adapter"):
    body = stage.GetPrimAtPath(f"/DynamixelArm/arm_carrier/{name}")
    body.GetAttribute("physics:mass").Set(0.001)
print("set arm_carrier/tool_adapter mass=0.001kg (was 0.0, invalid)", flush=True)

stage.GetRootLayer().Save()

print("=== resulting USD prim tree ===", flush=True)
for p in stage.Traverse():
    print(p.GetPath(), p.GetTypeName(), flush=True)

app.close()
