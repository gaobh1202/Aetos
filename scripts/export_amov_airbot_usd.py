"""Bake amov_airbot_gripper and amov_airbot_dexterous into one standalone USD.

Spawns both AerialManipulatorBase presets (see scripts/show_airbot_variants.py,
which this is a headless/export variant of) directly under the pseudo-root --
no ground plane, no lights, no PhysicsScene -- then flattens the whole stage
with Usd.Stage.Flatten(), which resolves every reference/payload/variant arc
into concrete prims. The result has no dependency on the original source USD
files (airbot_play_gripper.usd, airbot_play_bare.usd, inspire_hand.usda, the
drone USD, ...): it is the two fully-assembled robots as one self-contained
asset. (Flatten does not inline *texture* asset-path attributes -- only
composition arcs -- so any material textures referenced by relative path
would still need to travel alongside this file; none of the two variants use
textured materials today, so this isn't exercised, just noted for later.)

No sim.reset()/initialize() is needed: AerialManipulatorBase.spawn() authors
every prim/joint directly via mount_child at spawn time (robot_base.py) --
initialize() only resolves DOF indices into a running ArticulationView, it
doesn't write anything to the stage.

Output: aetos_multiam/assets/meshes/usd/combined/amov_airbot_combined.usd, with two
top-level robot prims: /amov_airbot_gripper and /amov_airbot_dexterous.

Run with the same interpreter as this repo's other scripts/run_*.py scripts:

    python scripts/export_amov_airbot_usd.py
"""
import logging
import os

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext

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

GRIPPER_PATH = "/amov_airbot_gripper"
DEXTEROUS_PATH = "/amov_airbot_dexterous"

am_gripper = AerialManipulatorBase(get_preset("amov950_airbot_bottom"), name="amov_airbot_gripper")
am_dexterous = AerialManipulatorBase(
    get_preset("amov950_airbot_dexterous_bottom"), name="amov_airbot_dexterous"
)

# Same 2.5m separation as show_airbot_variants.py, just so the two don't
# overlap in the exported file -- purely cosmetic, callers reposition freely.
am_gripper.spawn(translations=[(-1.25, 0.0, 0.0)], prim_paths=[GRIPPER_PATH])
am_dexterous.spawn(translations=[(1.25, 0.0, 0.0)], prim_paths=[DEXTEROUS_PATH])

stage = prim_utils.get_current_stage()

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "amov_airbot_combined.usd")

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened combined USD to %s", out_path)
logging.info("Root prims: %s, %s", GRIPPER_PATH, DEXTEROUS_PATH)

simulation_app.close()
