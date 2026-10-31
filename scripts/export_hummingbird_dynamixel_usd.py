"""Bake hummingbird340_dynamixel2r_bottom into one standalone USD, same
pattern as scripts/export_amov_airbot_usd.py: spawn() only (no
sim.reset()/initialize() -- AerialManipulatorBase.spawn() authors every
prim/joint directly via mount_child, so the static rest pose is already
correct at that point), then Usd.Stage.Flatten() to resolve every
reference/payload arc into concrete prims, producing one self-contained
asset with no dependency on the drone/arm/gripper source USD files.

Output: aetos_multiam/assets/meshes/usd/combined/hummingbird_dynamixel_combined.usd,
with one top-level robot prim: /hummingbird_dynamixel.
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

ROBOT_PATH = "/hummingbird_dynamixel"
am = AerialManipulatorBase(get_preset("hummingbird340_dynamixel2r_bottom"), name="hummingbird_dynamixel")
am.spawn(translations=[(0.0, 0.0, 0.0)], prim_paths=[ROBOT_PATH])

stage = prim_utils.get_current_stage()

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "hummingbird_dynamixel_combined.usd")

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened combined USD to %s", out_path)
logging.info("Root prim: %s", ROBOT_PATH)

simulation_app.close()
