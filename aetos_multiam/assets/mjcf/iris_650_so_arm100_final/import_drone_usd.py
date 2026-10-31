"""Import iris_650_drone.xml (see split_drone.py) via Isaac Sim's own
omni.importer.mjcf -- a free-floating 4-rotor body, no joints besides the
MJCF's own <freejoint>, so this is the same "importer handles it cleanly"
case as every other MJCF-only bundle in this project.

Unlike the arm imports elsewhere in this project (set_fix_base(True), later
stripped by mount_child's own weld), this is a genuinely free-floating
drone body -- set_fix_base(False) so the importer doesn't add a redundant
world anchor on top of the MJCF's own <freejoint>.

The MJCF's root body is named "iris_650" (matching the source bundle's own
body name), but every DroneSpec in this project expects mount_body_name=
"base_link" -- renamed post-import to match.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "iris_650_drone.xml"
OUT_DIR = HERE.parent.parent / "meshes" / "usd"
DEST = OUT_DIR / "iris_650.usd"

from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from pxr import Usd, UsdPhysics, Sdf

OUT_DIR.mkdir(parents=True, exist_ok=True)

ctx = omni.usd.get_context()
ctx.new_stage()

status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
import_config.set_fix_base(False)
import_config.set_import_inertia_tensor(True)

result = omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path=str(MJCF_PATH),
    import_config=import_config,
    prim_path="/Iris650",
    dest_path=str(DEST),
)
print("import result:", result, flush=True)

stage = Usd.Stage.Open(str(DEST))

print("=== prim tree right after import ===", flush=True)
for p in stage.Traverse():
    print(p.GetPath(), p.GetTypeName(), flush=True)

# The importer nests the MJCF's own root body one level under whatever
# prim_path wrapper is given (confirmed project-wide, doesn't change with
# the wrapper name) -- here that lands everything at
# /Iris650/iris_650/{iris_650 (the body itself, same name as its own
# wrapper -- MJCF root body here is literally named "iris_650"), rotor_0..3,
# joints}. Every OTHER drone in this project (amov_950, omnidrones_340/430)
# has its stage's defaultPrim resolve straight to a single "base_link" with
# no extra wrapper segment, and DroneSpec.mount_body_name="base_link" (a
# single path segment) everywhere assumes that -- rather than introduce a
# two-segment mount_body_name just for this drone, promote the wrapper's
# children (base_link/rotor_0..3/joints) up one level to /Iris650 directly
# and delete the now-empty intermediate wrapper, matching every other
# drone's layout exactly.
#
# Plain Sdf.CopySpec does NOT rewrite relationship targets (the joints'
# own body0/body1 rels point at sibling bodies by absolute path) -- use
# Sdf.BatchNamespaceEdit/Reparent instead, which retargets internal
# relationships as part of the same atomic edit.
layer = stage.GetRootLayer()
wrapper_path = Sdf.Path("/Iris650/iris_650")
body_path = wrapper_path.AppendChild("iris_650")

edit = Sdf.BatchNamespaceEdit()
for child in list(stage.GetPrimAtPath(wrapper_path).GetChildren()):
    old = child.GetPath()
    new_name = "base_link" if old == body_path else child.GetName()
    new = Sdf.Path("/Iris650").AppendChild(new_name)
    edit.Add(Sdf.NamespaceEdit(old, new))
    print(f"queued move {old} -> {new}", flush=True)
applied = layer.Apply(edit)
print("namespace edit applied:", applied, flush=True)
if stage.GetPrimAtPath(wrapper_path).IsValid():
    stage.RemovePrim(wrapper_path)
    print(f"removed now-empty wrapper {wrapper_path}", flush=True)

# Sdf.BatchNamespaceEdit/layer.Apply() moved the prims but did NOT rewrite
# the joints' own body0/body1 relationship targets (confirmed empirically:
# they were left pointing at the old pre-move paths) -- fix those by hand.
for rname in ("rotor_0", "rotor_1", "rotor_2", "rotor_3"):
    joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"/Iris650/joints/{rname}"))
    joint.GetBody0Rel().SetTargets([Sdf.Path("/Iris650/base_link")])
    joint.GetBody1Rel().SetTargets([Sdf.Path(f"/Iris650/{rname}")])
print("fixed up joint body0/body1 relationship targets", flush=True)

# set_fix_base(False) still left an empty, unconnected "worldBody" scaffold
# prim carrying ArticulationRootAPI (same importer quirk documented for the
# arm imports elsewhere in this project, just not stripped by set_fix_base
# this time) -- remove it and put ArticulationRootAPI on base_link directly,
# matching every other free-floating drone in this project.
world_body = stage.GetPrimAtPath("/Iris650/worldBody")
if world_body.IsValid():
    stage.RemovePrim(world_body.GetPath())
    print("removed empty worldBody scaffold prim", flush=True)
UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/Iris650/base_link"))
print("applied ArticulationRootAPI to /Iris650/base_link", flush=True)

stage.GetRootLayer().Save()

print("=== prim tree after rename ===", flush=True)
for p in stage.Traverse():
    print(p.GetPath(), p.GetTypeName(), flush=True)

app.close()
