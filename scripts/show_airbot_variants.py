"""Open an Isaac Sim GUI scene showing airbot_gripper and airbot_dexterous
side by side, for visual inspection of the new USD surgery (bare arm,
custom_end_link mount, Inspire Hand orientation) -- not a task/env, no
GridCloner/batched envs, just two AerialManipulatorBase instances spawned
directly into one stage via the amov950_airbot_bottom /
amov950_airbot_dexterous_bottom presets.

Gravity is disabled scene-wide (zeroed on the PhysicsScene prim, same idea
as `<option gravity="0 0 0">` in the MuJoCo combined-scene example) --
this is for looking at geometry/mounting, nothing here is meant to fly or
settle under gravity.

Run with the same interpreter this repo's other scripts/run_*.py scripts
use (Isaac Sim's own python.sh, or the omnidrone conda env):

    python scripts/show_airbot_variants.py

Saves snapshots (wide shot + one close-up per arm) to scripts/, via the
same get_active_viewport()/capture_viewport_to_file() pattern already used
by assets/usd/three_finger_fin_ray_rigid/render_bend_video_isaac.py, then
closes cleanly -- in this environment the Kit window has closed itself
~15-18s after spawning on every run so far (survived fixing two unrelated
scene bugs along the way, so it isn't either of those), so capture-then-
close is the reliable path rather than relying on a "stay open for live
inspection" loop. Root cause not fully isolated; if it needs a real
interactive session later, try `isaac-sim.sh` directly instead of the
`SimulationApp` Python wrapper.
"""
import logging
import os

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": False})

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
    physics_prim_path="/physicsScene",
    device="cuda:0",
)

kit_utils.create_ground_plane(
    "/World/defaultGroundPlane",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.8,
    improve_patch_friction=True,
)
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))

am_gripper = AerialManipulatorBase(get_preset("amov950_airbot_bottom"), name="airbot_gripper_demo")
am_dexterous = AerialManipulatorBase(get_preset("amov950_airbot_dexterous_bottom"), name="airbot_dexterous_demo")

GRIPPER_PRIM_PATH = "/World/airbot_gripper_demo"
DEXTEROUS_PRIM_PATH = "/World/airbot_dexterous_demo"

am_gripper.spawn(translations=[(-1.0, 0.0, 1.0)], prim_paths=[GRIPPER_PRIM_PATH])
am_dexterous.spawn(translations=[(1.0, 0.0, 1.0)], prim_paths=[DEXTEROUS_PRIM_PATH])

# Static side-by-side viewer, not a flight sim -- zero gravity so neither
# robot falls/drifts once physics starts stepping below.
physics_scene_path = sim.get_physics_context().prim_path
UsdPhysics.Scene(prim_utils.get_prim_at_path(physics_scene_path)).CreateGravityMagnitudeAttr(0.0)

sim.reset()

am_gripper.initialize(prim_paths_expr=GRIPPER_PRIM_PATH)
am_dexterous.initialize(prim_paths_expr=DEXTEROUS_PRIM_PATH)

logging.info(
    "Spawned airbot_gripper at x=-1.0 and airbot_dexterous at x=+1.0 "
    "(both on amov_950, gravity off)."
)

vp = get_active_viewport()
out_dir = os.path.dirname(os.path.abspath(__file__))


def snapshot(name, eye, target):
    set_camera_view(eye=eye, target=target)
    for _ in range(10):
        sim.render()
        simulation_app.update()
    path = os.path.join(out_dir, name)
    capture_viewport_to_file(vp, path)
    simulation_app.update()  # let the async capture flush to disk
    logging.info("Saved %s", path)


# Both arms hang ~0.125m below base_link (z=1.0) once mounted correctly --
# see manipulators/airbot_play.py's MOUNT_TRANSLATION/MOUNT_ORIENTATION.
snapshot("airbot_variants_preview.png", eye=[0.0, -3.0, 1.1], target=[0.0, 0.0, 0.6])
snapshot("airbot_gripper_closeup.png", eye=[-1.0, -1.2, 0.85], target=[-1.0, 0.0, 0.55])
snapshot("airbot_dexterous_closeup.png", eye=[1.0, -1.2, 0.85], target=[1.0, 0.0, 0.55])

simulation_app.close()
