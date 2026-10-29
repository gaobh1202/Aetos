# Examples

Runnable demos, no Isaac Sim required -- MuJoCo only. Run from the repo
root (paths are resolved relative to this file's location, not the
current working directory):

```bash
python examples/00_load_each_aerial_manipulator_mujoco.py
python examples/01_combine_all_aerial_manipulators_mujoco.py
```

- **`00_load_each_aerial_manipulator_mujoco.py`** -- loads each of this
  project's 5 aerial-manipulator MJCF bundles one at a time, confirms each
  compiles and steps, and writes a 2x3 grid preview
  (`all_five_preview.png`).
- **`01_combine_all_aerial_manipulators_mujoco.py`** -- merges all 5 into
  **one** MuJoCo scene (a namespace-prefixing XML merger, since the
  bundles reuse generic names like `link1`/`joint1` across each other and
  can't just be `<include>`d together) and renders it
  (`all_five_combined_scene.png` + the merged `all_five_combined.xml`
  itself, if you want to load it directly). See that script's own
  docstring for the real MJCF/MuJoCo pitfalls hit building it (dm_control
  schema gaps, `<compiler meshdir>`, keyframe sizing, free-joint nesting,
  a robot hidden behind the floor plane) -- worth a read before writing a
  similar multi-robot merge script.

Both scripts write their output next to themselves in this directory. The
preview PNGs are committed so they're visible straight from the repo
without running anything; the merged `all_five_combined.xml` is
regenerated output (git-ignored) -- re-run the script to get a fresh one.
