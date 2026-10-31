"""Spawn the bare iris_650 drone (no arm) through the real OmniDrones
Iris650Drone class + iris_650.yaml, exactly as AerialManipulatorBase would
use it -- confirms the freshly-imported USD actually works end-to-end with
RotorGroup/MultirotorBase, not just that the raw USD structure looks right."""
import logging

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdGeom

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.drones.iris_650 import IRIS_650

logging.basicConfig(level=logging.INFO)

sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 100,
    backend="torch", physics_prim_path="/World/PhysicsScene", device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))

PRIM_PATH = "/World/iris_650"
drone = IRIS_650.drone_class(name="iris_650")
drone.spawn(translations=[(0.0, 0.0, 1.0)], prim_paths=[PRIM_PATH])
sim.reset()
drone.initialize(prim_paths_expr=PRIM_PATH)
sim.reset()

print("num_rotors:", drone.num_rotors, flush=True)
print("mass:", drone.masses, flush=True)
print("inertia:", drone.inertias, flush=True)

stage = prim_utils.get_current_stage()
cache = UsdGeom.XformCache()
base_link = stage.GetPrimAtPath(f"{PRIM_PATH}/base_link")


def report(tag):
    cache.Clear()
    xf = cache.GetLocalToWorldTransform(base_link)
    pos = xf.ExtractTranslation()
    quat = xf.ExtractRotationQuat()
    print(f"[{tag}] base_link pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
          f"quat_w={quat.GetReal():+.4f} quat_xyz=({quat.GetImaginary()[0]:+.4f},"
          f"{quat.GetImaginary()[1]:+.4f},{quat.GetImaginary()[2]:+.4f})", flush=True)


report("t=0")
for i in range(100):
    sim.step(render=False)
    if i in (0, 99):
        report(f"t={(i+1)*0.01:.2f}s")

vp = get_active_viewport()
import os
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


snapshot("iris650_bare_iso.png", eye=(1.8, -1.8, 3.2), target=(0.0, 0.0, 2.5))

simulation_app.close()
