"""Headless: spawn airbot_dexterous and measure the real gap between
link5's and r_base_link's bounding boxes, plus which way is "up" for the
palm -- to fix a reported visible gap and a palm-up/palm-down mixup.
"""
from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Usd, UsdGeom

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

PRIM_PATH = "/World/debug_gap_demo"

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/physicsScene",
    device="cuda:0",
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")

am = AerialManipulatorBase(get_preset("amov950_airbot_dexterous_bottom"), name="debug_gap_demo")
am.spawn(translations=[(0.0, 0.0, 1.0)], prim_paths=[PRIM_PATH])
sim.reset()
am.initialize(prim_paths_expr=PRIM_PATH)

stage = prim_utils.get_current_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

link5 = stage.GetPrimAtPath(f"{PRIM_PATH}/manipulator/link5")
rbase = stage.GetPrimAtPath(f"{PRIM_PATH}/manipulator/gripper/r_base_link")

b5 = cache.ComputeWorldBound(link5).ComputeAlignedRange()
brb = cache.ComputeWorldBound(rbase).ComputeAlignedRange()
print("link5 bbox min/max:", tuple(round(v, 4) for v in b5.GetMin()), tuple(round(v, 4) for v in b5.GetMax()), flush=True)
print("r_base_link bbox min/max:", tuple(round(v, 4) for v in brb.GetMin()), tuple(round(v, 4) for v in brb.GetMax()), flush=True)

l5_xf = UsdGeom.Xformable(link5).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
rb_xf = UsdGeom.Xformable(rbase).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
print("link5 world pos:", tuple(round(v, 4) for v in l5_xf.ExtractTranslation()), flush=True)
print("r_base_link world pos:", tuple(round(v, 4) for v in rb_xf.ExtractTranslation()), flush=True)
q = rb_xf.ExtractRotationQuat()
print("r_base_link world quat(w,x,y,z):", round(q.GetReal(), 4), tuple(round(v, 4) for v in q.GetImaginary()), flush=True)

# which world direction does the hand's local +Y map to (palm normal guess)?
for axis_name, axis in [("local X", (1, 0, 0)), ("local Y", (0, 1, 0)), ("local Z (fingers)", (0, 0, 1))]:
    d = rb_xf.TransformDir((axis[0], axis[1], axis[2]))
    print(f"r_base_link {axis_name} -> world dir {tuple(round(v,4) for v in d)}", flush=True)

simulation_app.close()
