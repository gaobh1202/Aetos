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
      combined/                        amov_airbot_combined.usd (see scripts/export_amov_airbot_usd.py)
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

## Regenerating the combined file

`aetos_multiam/assets/meshes/usd/combined/amov_airbot_combined.usd` isn't
hand-authored -- it's produced by spawning both `amov950_airbot_bottom` and
`amov950_airbot_dexterous_bottom` presets and flattening the stage:

```bash
python scripts/export_amov_airbot_usd.py
```

Re-run it any time `airbot_play.py`'s mount offsets or `inspire_hand.py`'s
spec change, then re-upload the regenerated file to Google Drive.

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
