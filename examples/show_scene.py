"""Open an Isaac Sim window showing one arena, with no robot in it.

The counterpart to ``show_robot.py``: that one spawns a robot on a bare
ground plane, this one spawns a scene and nothing else -- for checking
layout, dimensions and lighting before any task or robot is wired up.

Usage:
    python examples/show_scene.py --scene default
    python examples/show_scene.py --scene warehouse
    python examples/show_scene.py --scene warehouse --headless   # screenshot only

Run via the omnidrone conda env (after `conda activate omnidrone`, with
ISAACSIM_PATH exported), not Isaac Sim's own python.sh.
"""
import argparse

from aetos_multiam import init_simulation_app

# name -> (arena class, camera position, camera target, focal length mm).
# Kit's persp camera ships an 18.1 mm lens on a 20.955 mm aperture, which is
# far too long to see a whole room from inside it -- hence the per-scene focal.
SCENES = {
    "default": ("Arena", (6.0, -8.0, 4.5), (0.0, 0.0, 0.5), 18.0),
    "warehouse": ("WarehouseArena", (7.6, -8.6, 6.2), (0.0, 0.2, 1.4), 10.0),
    "wall": ("WallArena", (4.0, -4.5, 3.0), (1.5, 0.0, 1.0), 18.0),
    "shelf": ("ShelfArena", (2.2, -2.6, 1.9), (0.0, 0.0, 0.8), 18.0),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=str, default="warehouse", choices=sorted(SCENES),
                    help="Which arena to spawn.")
parser.add_argument("--headless", action="store_true", help="Screenshot only, no GUI window.")
parser.add_argument("--screenshot", type=str, default=None,
                    help="Where to save the --headless screenshot (default: next to this script).")
parser.add_argument("--focal", type=float, default=None,
                    help="Camera focal length in mm; overrides the scene's own.")
parser.add_argument("--floor", choices=["grid", "concrete"], default="grid",
                    help="warehouse only: AM-Bench's grid plane, or a plain concrete slab.")
args = parser.parse_args()

simulation_app = init_simulation_app({"headless": args.headless})

import carb.settings
import omni.isaac.core.utils.prims as prim_utils
import omni.kit.commands
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

import aetos_multiam.scenes as scenes

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/World/PhysicsScene",
    device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, -9.81)},
)

# Kit draws its own reference grid (and the red/green world axes) over the
# viewport. On a floor it reads as texture that is not in the scene, which is
# exactly the wrong thing when the point of this script is to judge the floor.
carb.settings.get_settings().set("/app/viewport/grid/enabled", False)

try:
    cls_name, cam_pos, cam_target, focal = SCENES[args.scene]
    arena = getattr(scenes, cls_name)()
    if args.scene == "warehouse" and args.floor == "concrete":
        arena.floor_color = (0.70, 0.70, 0.68)
    arena.spawn("/World")

    # The ground plane brings its own DistantLight; a dome on top of it is
    # what stops the walls' inward faces from going flat black.
    prim_utils.create_prim("/World/Light/Dome", "DomeLight",
                           attributes={"inputs:intensity": 900.0})

    sim.reset()
    print(f"[show_scene] spawned {cls_name}", flush=True)

    vp = get_active_viewport()
    cam_path = vp.camera_path.pathString
    cam_state = ViewportCameraState(cam_path, vp)
    cam_state.set_position_world(Gf.Vec3d(*cam_pos), True)
    cam_state.set_target_world(Gf.Vec3d(*cam_target), True)
    # Via ChangeProperty, not prim.GetAttribute().Set(): Kit's persp camera
    # lives in the session layer, which outranks whatever edit target a plain
    # USD write lands on -- the write succeeds and changes nothing.
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=f"{cam_path}.focalLength",
        value=float(args.focal if args.focal is not None else focal),
        prev=None,
    )
    print(f"[show_scene] camera {cam_path} focal "
          f"{prim_utils.get_prim_at_path(cam_path).GetAttribute('focalLength').Get()}", flush=True)

    if args.headless:
        for _ in range(20):
            sim.render()
            simulation_app.update()
        out_path = args.screenshot or f"examples/show_scene_{args.scene}.png"
        capture_viewport_to_file(vp, out_path)
        simulation_app.update()
        print(f"[show_scene] saved {out_path}", flush=True)
    else:
        while simulation_app.is_running():
            sim.step(render=True)
except Exception:
    # Kit's sys.excepthook otherwise swallows this silently.
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
