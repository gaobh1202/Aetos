"""Interactive Isaac Sim window for iris650_soarm100_bottom (scaled Iris
650 + SO101-substitute SO-ARM100, mounted below, gap-fixed).

Inspection only (not a flight sim) -- same setup as
view_hummingbird_dynamixel.py:
- ``backend="torch"`` required (OmniDrones ArticulationView calls
  ``Tensor.unflatten``).
- ``device="cpu"`` keeps PhysX off the GPU Direct API, so viewport gizmos
  (move/rotate handles) can write poses.
- Gravity zeroed so the free-floating drone holds its spawn pose for manual
  inspection/editing.
"""
import logging

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": False})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, UsdPhysics

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
    device="cpu",
    sim_params={
        "use_gpu": False,
        "use_gpu_pipeline": False,
        "gravity": (0.0, 0.0, 0.0),
    },
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))
prim_utils.create_prim("/World/Light/GreySphere2", "SphereLight", translation=(-2.0, -2.0, 5.0))

PRIM_PATH = "/World/iris650_soarm100"
am = AerialManipulatorBase(get_preset("iris650_soarm100_bottom"), name="iris650_soarm100")
am.spawn(translations=[(0.0, 0.0, 1.0)], prim_paths=[PRIM_PATH])

physics_scene_path = sim.get_physics_context().prim_path
UsdPhysics.Scene(prim_utils.get_prim_at_path(physics_scene_path)).CreateGravityMagnitudeAttr(0.0)

sim.reset()
try:
    am.initialize(prim_paths_expr=PRIM_PATH)
    logging.info("Initialized %s", PRIM_PATH)
except Exception:
    logging.exception(
        "initialize failed; keeping the window open for visual inspection"
    )

logging.info(
    "Spawned %s on CPU PhysX, gravity=0. base_link sits at the MJCF's own "
    "authored z=1.5 + the 1.0 spawn offset = ~2.5m -- if the drone looks "
    "off-screen on open, scroll/orbit out.",
    PRIM_PATH,
)

vp = get_active_viewport()
cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
cam_state.set_position_world(Gf.Vec3d(1.6, -1.6, 2.9), True)
cam_state.set_target_world(Gf.Vec3d(0.0, 0.0, 2.3), True)

while simulation_app.is_running():
    sim.step(render=True)
    simulation_app.update()

simulation_app.close()
