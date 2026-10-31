"""Same bake-to-flattened-USD approach as export_amov_airbot_usd.py, but one
robot per file instead of two robots in one combined file. Run once per
preset (each run needs its own Isaac Sim process/stage, so this takes a
--preset/--name/--out argument instead of doing both in one pass):

    python scripts/export_amov_airbot_usd_split.py \
        --preset amov950_airbot_bottom --name amov_airbot_gripper \
        --out amov_airbot_gripper.usd

    python scripts/export_amov_airbot_usd_split.py \
        --preset amov950_airbot_dexterous_bottom --name amov_airbot_dexterous \
        --out amov_airbot_dexterous.usd

Each robot is spawned at the origin (no need to offset -- each file has
only the one robot) under its own top-level prim path /<name>.
"""
import argparse
import logging
import os

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser()
parser.add_argument("--preset", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

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

ROOT_PATH = f"/{args.name}"
am = AerialManipulatorBase(get_preset(args.preset), name=args.name)
am.spawn(translations=[(0.0, 0.0, 0.0)], prim_paths=[ROOT_PATH])

stage = prim_utils.get_current_stage()

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, args.out)

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened USD to %s", out_path)
logging.info("Root prim: %s", ROOT_PATH)

simulation_app.close()
