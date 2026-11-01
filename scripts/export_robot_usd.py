"""Bake one or more aerial-manipulator presets into one standalone, flattened
USD file -- config-driven generalization of what used to be five separate
one-off scripts (export_amov_airbot_usd.py, export_amov_airbot_usd_split.py,
export_hummingbird_dynamixel_usd.py, export_iris650_soarm100_usd.py,
export_fahexa750_usd.py).

Spawns every robot listed in the config, then flattens the whole stage with
Usd.Stage.Flatten(), which resolves every reference/payload/variant arc into
concrete prims -- the result has no dependency on the original source USD
files. No sim.reset()/initialize() is needed: AerialManipulatorBase.spawn()
authors every prim/joint directly via mount_child at spawn time, so the
static rest pose is already correct right after spawn().

Config is a YAML file (see scripts/export_configs/*.yaml for real examples):

    output: <filename>.usd      # written to aetos_multiam/assets/meshes/usd/combined/
    robots:
      - name: <unique top-level prim name>
        translation: [x, y, z]  # optional, default [0, 0, 0]
        preset: <preset name>              # XOR raw_usd_path, not both
        raw_usd_path: <module>:<ATTR_NAME> # for robots with no RotorGroup-compatible
                                            # preset yet (e.g. fahexa750/uaquad600's
                                            # drones) -- resolved by importing <module>
                                            # and reading its <ATTR_NAME> constant, not
                                            # duplicated as a literal path in the config

A config can list multiple robots (they get spawned side by side, each under
its own /<name> prim) to bake a combined multi-robot demo file, or just one
for a single standalone robot file.

Usage:
    python scripts/export_robot_usd.py --config scripts/export_configs/iris650_soarm100.yaml

Not covered here: uaquad600_ambencharm_top, which needs bespoke arm
re-mount surgery (reposing 8 bodies + patching a joint's local frame) on top
of the spawn -- a one-time fix for one specific asset, not something a
declarative config is a good fit for. See export_uaquad600_usd.py, kept as
its own script.
"""
import argparse
import importlib
import logging
import os

import yaml

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, help="Path to a YAML config (see module docstring).")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

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


def resolve_raw_usd_path(spec: str) -> str:
    """``spec`` is ``"<module>:<ATTR_NAME>"`` -- import the module and read
    the constant, rather than letting the config duplicate a literal
    filesystem path that could drift from the real DroneSpec's own."""
    module_name, attr = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


root_paths = []
for robot_cfg in cfg["robots"]:
    name = robot_cfg["name"]
    translation = tuple(robot_cfg.get("translation", [0.0, 0.0, 0.0]))
    root_path = f"/{name}"
    has_preset = "preset" in robot_cfg
    has_raw_path = "raw_usd_path" in robot_cfg
    if has_preset == has_raw_path:
        raise ValueError(
            f"Robot config {robot_cfg!r} must set exactly one of 'preset' or 'raw_usd_path'."
        )
    if has_preset:
        am = AerialManipulatorBase(get_preset(robot_cfg["preset"]), name=name)
        am.spawn(translations=[translation], prim_paths=[root_path])
    else:
        usd_path = resolve_raw_usd_path(robot_cfg["raw_usd_path"])
        prim_utils.create_prim(root_path, usd_path=usd_path, translation=translation)
    root_paths.append(root_path)

stage = prim_utils.get_current_stage()

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, cfg["output"])

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened USD to %s", out_path)
logging.info("Root prim(s): %s", root_paths)

simulation_app.close()
