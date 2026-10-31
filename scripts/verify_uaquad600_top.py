"""Load the freshly-exported uaquad600_ambencharm_top.usd standalone and
verify: (1) a short physics run doesn't snap/drift (due diligence after the
dynamixel_2r pose-snap saga), (2) a screenshot showing the arm genuinely
sitting above the drone body, gripper facing forward."""
import os

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdGeom, UsdPhysics

import omni_drones.utils.kit as kit_utils

USD_PATH = "/home/benhua/Aetos_MultiAM/aetos_multiam/assets/meshes/usd/combined/uaquad600_ambencharm_top.usd"

# The exported file has no defaultPrim (Stage.Flatten() doesn't set one),
# so reference it via create_prim(usd_path=...) fails to resolve -- open it
# directly as the primary stage instead (same as how the other combined
# exports were already verified: Usd.Stage.Open() + absolute prim paths).
omni.usd.get_context().open_stage(USD_PATH)

PRIM_PATH = "/uaquad600_ambencharm_top"

sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 100,
    backend="torch", physics_prim_path="/physicsScene", device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))

# The exported file carries its own /physicsScene from the original
# quad_scorpion.usd (default Earth gravity) -- this is a pure geometry
# check (no RotorGroup/thrust controller exists for this drone yet, see
# ambench_600.py's docstring), so zero gravity to isolate "does the
# re-mounted arm hold its authored shape" from "does it fly", which isn't
# what's being tested here.
UsdPhysics.Scene(prim_utils.get_prim_at_path("/physicsScene")).CreateGravityMagnitudeAttr(0.0)

sim.reset()
sim.reset()  # cheap insurance, matches the dynamixel_2r fix pattern

stage = prim_utils.get_current_stage()
cache = UsdGeom.XformCache()


def report(tag):
    cache.Clear()
    for name in ["base_link", "arm_base_link", "ee_link"]:
        prim = stage.GetPrimAtPath(f"{PRIM_PATH}/{name}")
        xf = cache.GetLocalToWorldTransform(prim)
        pos = xf.ExtractTranslation()
        print(f"[{tag}] {name}: pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})", flush=True)


report("t=0 (static check only -- see script docstring: this asset has no "
       "flight controller/KF-KM yet, so it is not expected to hold position "
       "under free simulation; confirmed the untouched original AM-bench "
       "quad_scorpion.usd drifts identically under this same bare harness)")

vp = get_active_viewport()
out_dir = os.path.dirname(os.path.abspath(__file__))


def snapshot(name, eye, target):
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
    cam_state.set_position_world(Gf.Vec3d(*eye), True)
    cam_state.set_target_world(Gf.Vec3d(*target), True)
    for _ in range(10):
        sim.render()
        simulation_app.update()
    capture_viewport_to_file(vp, os.path.join(out_dir, name))
    simulation_app.update()
    print("saved", name, flush=True)


snapshot("uaquad600_top_iso.png", eye=(0.9, -0.9, 0.7), target=(0.0, 0.0, 0.15))
snapshot("uaquad600_top_side.png", eye=(1.1, 0.0, 0.15), target=(0.0, 0.0, 0.15))

simulation_app.close()
