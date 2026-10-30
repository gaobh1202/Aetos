"""Headless diagnostic: spawn one aerial-manipulator preset and print the
world transform of its drone/arm/gripper bodies -- a quick way to check
"did the mount actually land where it should" without booting the full GUI
scene.

Grew out of debugging a real bug in ``utils/usd_compose.py:mount_child``
(see docs/ARCHITECTURE.md's "composable-mounting mechanism" section,
2026-10-30 follow-up): the arm was rendering roughly twice as high as it
should, and screenshots alone didn't make the cause obvious. Printing the
actual mount joint's ``localPos0``/``localRot0`` and comparing against
``parent_prim_path``'s own world pose is what found it -- kept here as a
reusable check for the next time a mount offset looks wrong.

    python scripts/debug_airbot_mount.py [preset_name]
"""
import sys

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Usd, UsdGeom, UsdPhysics

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

PRESET_NAME = sys.argv[1] if len(sys.argv) > 1 else "amov950_airbot_bottom"
PRIM_PATH = "/World/debug_mount_demo"

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/physicsScene",
    device="cuda:0",
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")

am = AerialManipulatorBase(get_preset(PRESET_NAME), name="debug_mount_demo")
am.spawn(translations=[(0.0, 0.0, 1.0)], prim_paths=[PRIM_PATH])
sim.reset()
am.initialize(prim_paths_expr=PRIM_PATH)

stage = prim_utils.get_current_stage()
root = stage.GetPrimAtPath(PRIM_PATH)

print(f"\n=== {PRESET_NAME}: world transforms ===", flush=True)
for p in Usd.PrimRange(root):
    if not p.HasAPI(UsdPhysics.RigidBodyAPI):
        continue
    xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = xf.ExtractTranslation()
    quat = xf.ExtractRotationQuat()
    print(
        f"{p.GetPath()}: pos={tuple(round(v, 4) for v in pos)} "
        f"quat(w,x,y,z)={(round(quat.GetReal(), 4), tuple(round(v, 4) for v in quat.GetImaginary()))}",
        flush=True,
    )

print("\n=== mount joints (anything named *MountJoint* or *FixedJoint*) ===")
for p in Usd.PrimRange(root):
    if "MountJoint" in p.GetName() or "FixedJoint" in p.GetName():
        j = UsdPhysics.Joint(p)
        print(
            f"{p.GetPath()} ({p.GetTypeName()}): body0={[str(t) for t in j.GetBody0Rel().GetTargets()]} "
            f"body1={[str(t) for t in j.GetBody1Rel().GetTargets()]} "
            f"localPos0={j.GetLocalPos0Attr().Get()} localPos1={j.GetLocalPos1Attr().Get()}",
            flush=True,
        )

simulation_app.close()
