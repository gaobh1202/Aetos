"""Open an interactive Isaac Sim window showing one aerial-manipulator preset,
or all of them side by side.

Generalizes what used to be separate one-off scripts
(scripts/show_airbot_variants.py, scripts/view_iris650_soarm100.py, etc.) --
one script, pick the preset by name.

Gravity is zeroed scene-wide; this is for looking at geometry/mounting, not
flight -- matches the convention every one of those older scripts used.

Usage:
    python examples/show_robot.py --robot iris650_soarm100_bottom
    python examples/show_robot.py --robot all
    python examples/show_robot.py --robot all --headless   # screenshot only

Run via the omnidrone conda env (after `conda activate omnidrone`, with
ISAACSIM_PATH exported so its activation script wires up Isaac Sim's own
PYTHONPATH -- see docs/TELEOP.md), not Isaac Sim's own python.sh (that
interpreter doesn't have aetos_multiam/omni_drones installed).
"""
import argparse

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--robot", type=str, default="all",
    help="Preset name from aetos_multiam.robots.presets.PRESETS, or 'all' to show every registered preset.",
)
parser.add_argument("--headless", action="store_true", help="Screenshot only, no GUI window.")
args = parser.parse_args()

simulation_app = init_simulation_app({"headless": args.headless})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdPhysics

import omni_drones.utils.kit as kit_utils
from aetos_multiam.assets import AssetNotAvailableError
from aetos_multiam.robots.presets import PRESETS, get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/World/PhysicsScene",
    device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
)
kit_utils.create_ground_plane("/World/defaultGroundPlane")
prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))
prim_utils.create_prim("/World/Light/GreySphere2", "SphereLight", translation=(-2.0, -2.0, 5.0))

try:
    names = sorted(PRESETS) if args.robot == "all" else [args.robot]

    SPACING = 2.0  # meters between robots when showing more than one
    spawned = []  # list of (name, prim_path, AerialManipulatorBase)
    for name in names:
        try:
            am = AerialManipulatorBase(get_preset(name), name=name)
        except (KeyError, AssetNotAvailableError) as e:
            print(f"[show_robot] skipping '{name}': {e}", flush=True)
            continue
        prim_path = f"/World/{name}"
        # Position by how many have actually spawned so far, not the index
        # into `names` -- otherwise a skipped preset leaves a gap and every
        # robot after it ends up further from the camera's framing than
        # expected.
        am.spawn(translations=[(len(spawned) * SPACING, 0.0, 1.0)], prim_paths=[prim_path])
        spawned.append((name, prim_path, am))

    if not spawned:
        print("[show_robot] no preset could be spawned -- nothing to show.", flush=True)
        raise SystemExit(1)

    physics_scene_path = sim.get_physics_context().prim_path
    UsdPhysics.Scene(prim_utils.get_prim_at_path(physics_scene_path)).CreateGravityMagnitudeAttr(0.0)

    sim.reset()
    for _, prim_path, am in spawned:
        am.initialize(prim_paths_expr=prim_path)

    print(f"[show_robot] spawned {len(spawned)}/{len(names)} requested preset(s): "
          f"{[n for n, _, _ in spawned]}", flush=True)

    vp = get_active_viewport()
    mid_x = (len(spawned) - 1) * SPACING / 2.0
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
    cam_state.set_position_world(Gf.Vec3d(mid_x + 1.6, -2.5, 2.9), True)
    cam_state.set_target_world(Gf.Vec3d(mid_x, 0.0, 1.0), True)

    if args.headless:
        for _ in range(10):
            sim.render()
            simulation_app.update()
        out_path = "examples/show_robot_preview.png"
        capture_viewport_to_file(vp, out_path)
        simulation_app.update()
        print(f"[show_robot] saved {out_path}", flush=True)
    else:
        while simulation_app.is_running():
            sim.step(render=True)
except Exception:
    # Kit's sys.excepthook otherwise swallows an uncaught exception silently
    # (clean exit, no traceback) -- print it explicitly.
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
