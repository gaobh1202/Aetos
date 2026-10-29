"""Load all 5 of this project's aerial-manipulator MJCF bundles in MuJoCo,
confirm each compiles and steps, and render a combined grid preview.

The 5 (matching robots/presets.py's PRESETS, one MJCF bundle per preset):
  1. amov950_airbot_bottom      -> amov950_airbot_final/
  2. hummingbird340_dynamixel2r_bottom -> hummingbird_2r_three_finger_50pct/
  3. iris650_soarm100_bottom    -> iris_650_so_arm100_final/
  4. uaquad600_ambencharm_top   -> ambench_quad_600_top_arm_final/
  5. fahexa750_ambencharm_top   -> ambench_fa_hexa_750_final/

4 of these 5 also have (partial) Isaac Sim USD assets today -- only
hummingbird+three-finger doesn't, because its gripper is a MuJoCo
`flexcomp` soft body with no PhysX equivalent (see
aetos_multiam/assets/mjcf/three_finger_fin_ray_rigid/DERIVATION.md for the
rigid approximation built instead). This script is MuJoCo-only, no Isaac
Sim involved.

Run from anywhere (paths are resolved relative to the repo root, not the
current working directory):

    python examples/00_load_each_aerial_manipulator_mujoco.py

See 01_combine_all_aerial_manipulators_mujoco.py for loading all 5 into
one shared scene instead of 5 separate renders.
"""
import os
import mujoco
import numpy as np
import imageio.v2 as imageio

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
BUNDLE_ROOT = os.path.join(REPO_ROOT, "aetos_multiam", "assets", "mjcf")

BUNDLES = [
    ("amov950_airbot_bottom", "amov950_airbot_final"),
    ("hummingbird340_dynamixel2r_bottom", "hummingbird_2r_three_finger_50pct"),
    ("iris650_soarm100_bottom", "iris_650_so_arm100_final"),
    ("uaquad600_ambencharm_top", "ambench_quad_600_top_arm_final"),
    ("fahexa750_ambencharm_top", "ambench_fa_hexa_750_final"),
]

tiles = []
for preset_name, bundle_dir in BUNDLES:
    scene_path = os.path.join(BUNDLE_ROOT, bundle_dir, "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0) if model.nkey > 0 else None
    mujoco.mj_forward(model, data)
    for _ in range(5):
        mujoco.mj_step(model, data)

    # Ignore any named in-model cameras (amov950_airbot_final's only one,
    # "eye_arm", is a close-up gripper-mounted camera, not a scene
    # overview -- unusable for a "does this robot load" preview). Build a
    # free camera from each scene's own <statistic center/extent>, matching
    # the azimuth/elevation every one of these scene.xml files already
    # authors in <visual><global>.
    renderer2 = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = model.stat.center
    cam.distance = model.stat.extent * 1.8
    cam.azimuth = 135
    cam.elevation = -20
    renderer2.update_scene(data, camera=cam)
    img = renderer2.render()
    renderer2.close()  # creating many Renderers in one process without
    # closing each one first left every render after the first blank/black
    # (an exhausted GL context, not a camera/geometry problem) -- found by
    # comparing xpos against stat.center (matched fine) before suspecting this.
    tiles.append(img)

    print(f"[OK] {preset_name:38s} <- {bundle_dir:34s} "
          f"nq={model.nq:3d} nbody={model.nbody:3d} njnt={model.njnt:2d} "
          f"nu={model.nu:2d}", flush=True)

# 2x3 grid (last cell blank)
h, w = tiles[0].shape[:2]
grid = np.full((h * 2, w * 3, 3), 235, dtype=np.uint8)
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
for (r, c), img in zip(positions, tiles):
    grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img[..., :3]

out_path = os.path.join(EXAMPLES_DIR, "all_five_preview.png")
imageio.imwrite(out_path, grid)
print("\nWrote combined preview:", out_path, flush=True)
print(f"All {len(tiles)}/5 aerial manipulators loaded and stepped successfully in MuJoCo.", flush=True)
