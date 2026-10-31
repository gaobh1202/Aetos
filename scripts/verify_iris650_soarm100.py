"""Spawn the full iris650_soarm100_bottom preset -- scaled Iris 650 drone +
Lightwheel SO101 (so_arm100 substitute) mounted below the body -- and
verify: (1) the mount_orientation just computed for so_arm100.py puts the
gripper in a sane spot (below the drone, not clipping through the frame),
(2) a short physics run (double sim.reset(), matching the dynamixel_2r
fix) doesn't snap/drift."""
import logging

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdGeom

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

logging.basicConfig(level=logging.INFO)

sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 100,
    backend="torch", physics_prim_path="/World/PhysicsScene", device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))

PRIM_PATH = "/World/iris650_soarm100"
am = AerialManipulatorBase(get_preset("iris650_soarm100_bottom"), name="iris650_soarm100")
am.spawn(translations=[(0.0, 0.0, 1.0)], prim_paths=[PRIM_PATH])
sim.reset()
am.initialize(prim_paths_expr=PRIM_PATH)
# robot_base.py's own extra reset (Update 18 fix) already fires inside
# initialize() when a manipulator is mounted -- no need to repeat it here.

stage = prim_utils.get_current_stage()
cache = UsdGeom.XformCache()


def report(tag):
    cache.Clear()
    for name, path in [
        ("base_link", f"{PRIM_PATH}/base_link"),
        ("manipulator_base", f"{PRIM_PATH}/manipulator/base"),
        ("manipulator_jaw", f"{PRIM_PATH}/manipulator/jaw"),
    ]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[{tag}] {name}: MISSING at {path}", flush=True)
            continue
        xf = cache.GetLocalToWorldTransform(prim)
        pos = xf.ExtractTranslation()
        quat = xf.ExtractRotationQuat()
        print(f"[{tag}] {name}: pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
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


snapshot("iris650_soarm100_iso.png", eye=(1.3, -1.3, 2.9), target=(0.0, 0.0, 2.3))
snapshot("iris650_soarm100_side.png", eye=(1.8, 0.0, 2.3), target=(0.0, 0.0, 2.3))
snapshot("iris650_soarm100_front.png", eye=(0.0, -1.8, 2.3), target=(0.0, 0.0, 2.3))

simulation_app.close()
