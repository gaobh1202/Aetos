# Model assets (not stored in git)

Every binary robot model file -- USD and every MJCF bundle's mesh geometry
(`.obj`/`.stl`), plus a couple of model demo videos -- lives under one
folder, `aetos_multiam/assets/meshes/`, and is **not** committed to this
repo's GitHub remote. It's distributed via Google Drive instead. The whole
folder is one `.gitignore` entry (`aetos_multiam/assets/meshes/`) -- nothing
else under `aetos_multiam/assets/` is ignored.

`check_asset()` (see `aetos_multiam/assets.py`) raises
`AssetNotAvailableError` with the expected path if a file a spec needs is
missing.

## Download

Google Drive link: https://drive.google.com/file/d/1FZKNlKgmVvU7OkbTPIkQhavGNXFxqqyX/view?usp=sharing

The archive's own top level is `usd/` and `mjcf/` directly (not wrapped in
an extra `meshes/` folder), so extract it *into* a `meshes/` directory you
create under `aetos_multiam/assets/`:

```bash
mkdir -p aetos_multiam/assets/meshes
tar -xzf aetos_multiam_meshes.tar.gz -C aetos_multiam/assets/meshes/
# or, for a .zip:
# unzip aetos_multiam_meshes.zip -d aetos_multiam/assets/meshes/
```

## Layout

```
aetos_multiam/assets/
  meshes/                              <- the one gitignored folder, Google Drive
    usd/
      airbot_play/                     airbot_play_gripper.usd, airbot_play_bare.usd
      inspire_hand/
        inspire_hand_g1dfq_left.usd
        inspire_hand_kuka_right.usd
        inspire_hand_right/            inspire_hand.usda + payloads/ + Materials/ + LICENSE
        inspire_hand_candidates/       raw extraction sources (g1_hands.usd, kuka_w_hand.usd)
      so101_follower/                  so101_follower.usd
      three_finger_fin_ray_rigid/      finger_1_rigid.usd + finger_1_bend_demo_isaacsim.mp4
      combined/                        amov_airbot_combined.usd + others (see scripts/export_robot_usd.py)
    mjcf/
      ambench_fa_hexa_750_final/assets/
      ambench_quad_600_top_arm_final/assets/
      amov950_airbot_final/assets/
      hummingbird_2r_three_finger_50pct/assets/
      iris_650_so_arm100_final/assets/
      three_finger_fin_ray_rigid/assets/ + finger_1_bend_demo.mp4
      three_finger_gripper_final/assets/
  usd/                                 <- stays tracked in git (small, real files)
    iris_650.yaml                      real RotorGroup-compatible rotor-physics config
    inspire_hand/extract_inspire_hands.py
    three_finger_fin_ray_rigid/{build_finger_1_rigid.py, render_bend_video_isaac.py}
  mjcf/<bundle>/                       <- stays tracked (robot.xml/scene.xml, README,
                                           CONVERSION_NOTES, preview thumbnails, launch.py, ...)
```

## Task-object assets (`objects/assets/`)

`aetos_multiam/objects/assets/{shelf_chrome,gearwheel,button,power_drill,
tote_box}/` holds the USD/mesh/texture binaries for the `AerialButtonGrasp`
task's props, migrated from raids-am-simulator's shelf_grasp scene. Same
not-committed-to-git convention as `assets/meshes/` above (`.gitignore`
entry: `aetos_multiam/objects/assets/`) -- copy the folder from
raids-am-simulator's `assets/objects/{shelf_chrome,gearwheel,button,
power_drill,tote_box}/` (or wherever it's redistributed from) if it's
missing. Each prop's Python wrapper (`aetos_multiam/objects/{gearwheel,
shelf_button,power_drill,tote_box}.py`) points at its own subfolder here via
a path relative to `objects/__init__.py`, not an absolute path into another
repo.

Three more vendored bundles live here too, from NVIDIA's SimReady asset
library (downloaded directly, not via raids-am-simulator):
`powerTool_drill_b01/`, `handTool_hammer_crossPeen_b01/` and
`attached_lid_tote/`. Each folder is that asset's own `simready_usd/`
payload as-is (root `.usd`/`.usda` + `payloads/`/`forgecad/` +
`Textures`/`assets` + thumbnails) plus its `LICENSE.txt` where the download
included one -- no `.cache/huggingface` download bookkeeping or
`.metadata/*.json` conformance reports carried over, those aren't asset
content. **No Python object wrapper exists for any of these yet** (no
`AetosObject` subclass, no task references them) -- they're vendored for
future use. Follow the `gearwheel.py`/`power_drill.py` pattern to add one:
point a new `AetosObject` subclass's `usd_path` at the root `.usd`/`.usda`
file here, and read that file's own body names for `root_body_name`.

`white_bucket/` is wired up already -- see `objects/white_bucket.py`
(`WhiteBucket`, used by `AerialBucketGrasp`). It's PalatialSim's
`palatial-nv-asset-library` dataset asset `D00436` rather than an NVIDIA
SimReady download, laid out the same way (`simready_usd/` + `.metadata/` +
`com.nvidia.simready.packaging.json`, no `.cache/huggingface` bookkeeping).

`aetos_multiam.assets.USD_PATH` (and everything derived from it --
`MESHES_PATH`, every manipulator/gripper spec's `usd_path`) points into
`assets/meshes/usd/`; each MJCF bundle's `robot.xml`/`scene.xml` resolves
its meshes via `<compiler meshdir="../../meshes/mjcf/<bundle>/assets">` (or,
for the one bundle whose `file=` values already carry their own `assets/`
prefix, `meshdir="../../meshes/mjcf/<bundle>">`) -- both point at the same
unified folder, just reached two different ways (Python path join vs. MJCF's
own relative-path resolution).

One exception: `aetos_multiam/assets/usd/iris_650.yaml` is a small,
hand-calibrated rotor-physics config, not a mesh/USD binary -- it stays
tracked in git at its own path, referenced directly via `ASSET_PATH`
(not `USD_PATH`) in `robots/drones/iris_650.py`.

## Regenerating the combined files

Every file under `combined/` except `uaquad600_gripper.usd` isn't
hand-authored -- it's produced by spawning one or more presets (or, for
drones with no RotorGroup-compatible param yaml yet, referencing their raw
source USD directly) and flattening the stage, driven by a YAML config:

```bash
python scripts/export_robot_usd.py --config scripts/export_configs/amov_airbot_combined.yaml
```

One config per output file under `scripts/export_configs/` -- see
`scripts/export_robot_usd.py`'s own module docstring for the config format.
Re-run the relevant one any time a mount offset or manipulator/gripper spec
changes, then re-upload the regenerated file to Google Drive.

`uaquad600_gripper.usd` is the one exception: it needs bespoke arm
re-mount surgery on top of the spawn (AM-bench's own source USD never
actually got re-mounted top-side-up, only its MJCF viewer copy did -- see
`scripts/export_uaquad600_usd.py`'s module docstring for the full story),
so it stays its own standalone script rather than a YAML config.

## The Factory gear set (downloadable, not on the netdisk)

`AerialAssembly` uses IsaacLab's own gear assets. They are **not** shipped
with Isaac Sim 4.1 -- its `Props/Factory/` has only nuts and bolts -- but they
exist in the 4.5 and 5.0 asset trees and load fine in 4.1. Fetch them into
`aetos_multiam/objects/assets/factory_gears/` (gitignored like every other
object asset):

```bash
mkdir -p aetos_multiam/objects/assets/factory_gears
cd aetos_multiam/objects/assets/factory_gears
BASE=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory
for f in factory_gear_base factory_gear_large factory_gear_medium factory_gear_small; do
    curl -O "$BASE/$f.usd"
done
```

Two properties of these files that cost real debugging time, both documented
in `aetos_multiam/objects/factory_gear.py`: each gear carries an
`ArticulationRootAPI` (so PhysX never registers it as a free rigid body until
that is stripped), and the colliders are SDF (so they have **no collision at
all** under a CPU PhysX pipeline).

## History note

Everything under `assets/meshes/` used to be split across
`aetos_multiam/assets/usd/` and each MJCF bundle's own `assets/` subfolder,
and some of those files used to be committed directly to this repo's GitHub
remote. They were stripped from every commit with
`git filter-repo --invert-paths` and the rewritten history force-pushed --
GitHub no longer has these bytes in any commit. One file,
`inspire_hand_g1dfq_right.usd`, was dropped entirely rather than kept
(retired -- superseded by `inspire_hand_right`, see
`grippers/inspire_hand.py`).
