"""Interactive Isaac Sim window for hummingbird340_dynamixel2r_bottom.

Inspection only (not a flight sim):
- ``backend="torch"`` is required: OmniDrones ArticulationView calls
  ``Tensor.unflatten``. ``backend="numpy"`` crashes at initialize.
- ``device="cpu"`` keeps PhysX off the GPU Direct API, so the viewport
  gizmo can write poses (``cuda:0`` enables ``eENABLE_DIRECT_GPU_API``).
- Gravity is zeroed so the free-floating drone holds its spawn pose.
- Camera is an explicit Z-up look-at from above, so the hummingbird X-frame
  reads as parallel to the ground (the airframe is only ~8 cm thick; a
  side view looks like a vertical line and is easy to misread).
"""
import logging

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": False})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdPhysics

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

logging.basicConfig(level=logging.INFO)

# torch tensors for OmniDrones; CPU PhysX so gizmos are legal.
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

# World-frame RGB axes at the origin so Z-up is obvious (Z = blue, up).
# Cube default extent is 2, so these scales make 25 cm x 1 cm bars.
for name, translation, scale, color in (
    ("X", (0.125, 0.0, 0.0), (0.125, 0.005, 0.005), (1.0, 0.2, 0.2)),
    ("Y", (0.0, 0.125, 0.0), (0.005, 0.125, 0.005), (0.2, 1.0, 0.2)),
    ("Z", (0.0, 0.0, 0.125), (0.005, 0.005, 0.125), (0.2, 0.4, 1.0)),
):
    prim = prim_utils.create_prim(
        f"/World/Axis{name}",
        "Cube",
        translation=translation,
        scale=scale,
    )
    prim.GetAttribute("primvars:displayColor").Set([color])

PRIM_PATH = "/World/hummingbird_dynamixel"
am = AerialManipulatorBase(get_preset("hummingbird340_dynamixel2r_bottom"), name="hummingbird_dynamixel")
am.spawn(translations=[(0.0, 0.0, 1.2)], prim_paths=[PRIM_PATH])

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
    "Spawned %s on CPU PhysX, gravity=0. "
    "From above the black X is the drone (parallel to the ground); "
    "the silver links hang in -Z. Side views of the 8 cm-thick airframe "
    "look like a vertical line -- that is not a 90 deg spawn error.",
    PRIM_PATH,
)

# Explicit Z-up look-at. set_camera_view() keeps /OmniverseKit_Persp's
# default Y-up, which makes a Z-up hummingbird look vertical in the GUI.
eye = Gf.Vec3d(0.6, -0.6, 2.6)
target = Gf.Vec3d(0.0, 0.0, 1.1)
up = Gf.Vec3d(0.0, 0.0, 1.0)
cam_prim = prim_utils.get_current_stage().GetPrimAtPath("/OmniverseKit_Persp")
look = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()
if cam_prim.IsValid() and cam_prim.GetAttribute("xformOp:orient"):
    q = look.ExtractRotationQuat()
    cam_prim.GetAttribute("xformOp:translate").Set(eye)
    cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(q))
else:
    vp = get_active_viewport()
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
    cam_state.set_position_world(eye, True)
    cam_state.set_target_world(target, True)

while simulation_app.is_running():
    sim.step(render=True)
    simulation_app.update()

simulation_app.close()
