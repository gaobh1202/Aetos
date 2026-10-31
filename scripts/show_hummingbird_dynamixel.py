"""Spawn the hummingbird340_dynamixel2r_bottom preset (Hummingbird drone +
dynamixel_2r arm + rigid three_finger_gripper) and screenshot it -- first
live test of this full combo assembly.
"""
import logging
import os

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import UsdPhysics

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

logging.basicConfig(level=logging.INFO)

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/World/PhysicsScene",
    device="cuda:0",
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))

PRIM_PATH = "/World/hummingbird_dynamixel"
am = AerialManipulatorBase(get_preset("hummingbird340_dynamixel2r_bottom"), name="hummingbird_dynamixel")
am.spawn(translations=[(0.0, 0.0, 1.2)], prim_paths=[PRIM_PATH])

physics_scene_path = sim.get_physics_context().prim_path
UsdPhysics.Scene(prim_utils.get_prim_at_path(physics_scene_path)).CreateGravityMagnitudeAttr(0.0)

sim.reset()
am.initialize(prim_paths_expr=PRIM_PATH)
logging.info("Spawned hummingbird340_dynamixel2r_bottom at %s", PRIM_PATH)

vp = get_active_viewport()
out_dir = os.path.dirname(os.path.abspath(__file__))


def snapshot(name, eye, target):
    set_camera_view(eye=eye, target=target)
    for _ in range(10):
        sim.render()
        simulation_app.update()
    path = os.path.join(out_dir, name)
    capture_viewport_to_file(vp, path)
    simulation_app.update()
    logging.info("Saved %s", path)


snapshot("hummingbird_dynamixel_wide.png", eye=[1.2, -1.2, 1.4], target=[0.0, 0.0, 0.8])
snapshot("hummingbird_dynamixel_close.png", eye=[0.4, -0.5, 1.2], target=[0.0, 0.0, 0.75])

simulation_app.close()
