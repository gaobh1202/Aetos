"""Assert that every preset's airframe wears the shared livery -- and that
nothing else does.

``AerialManipulatorBase.spawn`` re-binds each drone's frame to plastic_black
and its propellers to plastic_red (``utils.apply_drone_livery``). Which prims
count as "propeller" differs per asset -- separate ``rotor_*``/``rotor_link_*``
bodies on the OmniDrones and amov assets, a ``prop_mount_*`` subtree inside
``base_link`` on AM-bench's -- so this walks the spawned stage and checks the
classification actually landed, rather than trusting the name matching.

Checks, per preset:
  * every livery-bound prim sits under the drone's own base_link or a rotor
    body -- i.e. the manipulator/gripper (and amov's bundled-in arm, whose
    bodies are flat siblings of base_link) kept their own colours;
  * at least one frame prim and the expected number of propeller prims;
  * the two opted-out assets (omnidrones_340, ambench_750) are untouched.

No sim.reset(): spawn authors every prim directly, so the bindings exist as
soon as it returns.

Usage:
    python debug/check_drone_livery.py
"""
from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Usd, UsdGeom, UsdShade

from aetos_multiam.assets import AssetNotAvailableError
from aetos_multiam.robots.presets import PRESETS, get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase
from aetos_multiam.utils.usd_fixups import FRAME_MATERIAL, PROP_MATERIAL

# How many propeller *visual* prims each preset's drone should end up with.
EXPECTED_PROPS = {
    "amov950_airbot_bottom": 4,
    "amov950_airbot_dexterous_bottom": 4,
    "iris650_soarm100": 4,
    "uaquad600_gripper": 4,
}

sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 100,
    backend="torch", physics_prim_path="/World/PhysicsScene", device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False},
)

failures = []
try:
    for i, name in enumerate(sorted(PRESETS)):
        try:
            am = AerialManipulatorBase(get_preset(name), name=name)
        except (KeyError, AssetNotAvailableError) as e:
            failures.append(f"{name}: could not build preset ({e})")
            continue
        root_path = f"/World/{name}"
        am.spawn(translations=[(i * 3.0, 0.0, 1.0)], prim_paths=[root_path])

        root = prim_utils.get_prim_at_path(root_path)
        frame, prop, stray = [], [], []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not bound:
                continue
            kind = bound.GetPath().name
            if kind not in ("livery_frame", "livery_prop"):
                continue
            relative = prim.GetPath().pathString[len(root_path) + 1:]
            body = relative.split("/")[0]
            if body != "base_link" and "rotor" not in body and "prop" not in body:
                stray.append(relative)
            (prop if kind == "livery_prop" else frame).append(relative)

        if not am.drone_spec.livery:
            if frame or prop:
                failures.append(f"{name}: opted out of the livery but {len(frame) + len(prop)} prims wear it")
            print(f"  {name:34s} opted out, untouched  ok", flush=True)
            continue

        expected = EXPECTED_PROPS.get(name)
        if expected is None:
            failures.append(f"{name}: no expected propeller count recorded -- add one to EXPECTED_PROPS")
        elif len(prop) != expected:
            failures.append(f"{name}: {len(prop)} propeller prims re-bound, expected {expected}")
        if not frame:
            failures.append(f"{name}: no frame prim re-bound")
        if stray:
            failures.append(f"{name}: livery leaked outside the airframe: {stray}")
        print(f"  {name:34s} frame={len(frame):2d} ({FRAME_MATERIAL})  "
              f"props={len(prop):2d} ({PROP_MATERIAL})", flush=True)
    # Printed before close(): Kit's fast shutdown swallows anything after it.
    if failures:
        print("\nFAILED:", flush=True)
        for f in failures:
            print("  -", f, flush=True)
    else:
        print("\nDrone livery check PASSED.", flush=True)
finally:
    simulation_app.close()

raise SystemExit(1 if failures else 0)
