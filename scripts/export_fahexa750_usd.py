"""Bake fahexa750_ambencharm_top into one standalone USD.

Unlike uaquad600 (see export_uaquad600_usd.py), AM-bench's own
hexa_scorpion.usd already ships the arm top-mounted on the original design
("六旋翼保持原上置装配" -- see aetos_multiam/robots/manipulators/ambench_hexa_arm.py's
docstring) -- no re-mount surgery needed.

This references AM-bench's hexa_scorpion.usd directly via
prim_utils.create_prim() instead of going through
AerialManipulatorBase/MultirotorBase -- that path requires a
RotorGroup-compatible KF/KM params yaml (ambench_750.yaml) which doesn't
exist yet (AM-bench's own reduced-order RotorActuator has no equivalent
pair to port, and this project deliberately hasn't fabricated one -- see
aetos_multiam/robots/drones/ambench_750.py's docstring). A pure geometry
bake doesn't need RotorGroup actuation at all, so there's nothing lost by
skipping that layer here.

Output: aetos_multiam/assets/meshes/usd/combined/fahexa750_ambencharm_top.usd
"""
import logging
import os

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext

from aetos_multiam.robots.drones.ambench_750 import AMBENCH_750_USD_PATH

logging.basicConfig(level=logging.INFO)

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/physicsScene",
    device="cuda:0",
)

ROOT_PATH = "/fahexa750_ambencharm_top"
prim_utils.create_prim(ROOT_PATH, usd_path=AMBENCH_750_USD_PATH)

stage = prim_utils.get_current_stage()

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fahexa750_ambencharm_top.usd")

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened USD to %s", out_path)
logging.info("Root prim: %s", ROOT_PATH)

simulation_app.close()
