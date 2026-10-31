"""Bake iris650_soarm100_bottom into one standalone USD -- same
spawn()-then-Flatten() pattern as export_amov_airbot_usd_split.py /
export_hummingbird_dynamixel_usd.py (no sim.reset()/initialize() needed:
AerialManipulatorBase.spawn() authors every prim/joint directly via
mount_child at spawn time).

Output: aetos_multiam/assets/meshes/usd/combined/iris650_soarm100_bottom.usd
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

ROOT_PATH = "/iris650_soarm100_bottom"
am = AerialManipulatorBase(get_preset("iris650_soarm100_bottom"), name="iris650_soarm100_bottom")
am.spawn(translations=[(0.0, 0.0, 0.0)], prim_paths=[ROOT_PATH])

stage = prim_utils.get_current_stage()

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "iris650_soarm100_bottom.usd")

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened USD to %s", out_path)
logging.info("Root prim: %s", ROOT_PATH)

simulation_app.close()
