# Aetos_MultiAM

A benchmark for **multi aerial-manipulator (AM) collaboration**: one robot
*type* -- an aerial manipulator (drone + optional arm + optional gripper) --
composed from swappable drone / manipulator / gripper models, built on Isaac
Sim's `omni.isaac.core` API.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how this is organized
and how it maps onto its reference projects (robosuite, dexmimicgen,
OmniDrones, AM-bench), including an honest accounting of what's a real,
working asset vs. registered scaffolding.

## Status

This is a first-pass scaffold. **One task actually runs end-to-end:**
`ButtonGrasp` -- a 950mm AmovDrone + airbot_play arm flies up to a
wall-mounted button and presses it. Every other drone/manipulator/gripper
registered under `aetos_multiam/robots/` is a data spec + registry entry,
not yet exercised by a task (see the asset table in `docs/ARCHITECTURE.md`).

Six named aerial-manipulator presets exist (`aetos_multiam.robots.PRESETS` /
`robots/presets.py`), named `<drone><manipulator>_<mount>`:
`amov950_airbot_bottom`, `amov950_airbot_dexterous_bottom`,
`hummingbird340_dynamixel2r_bottom`, `iris650_soarm100_bottom`,
`uaquad600_ambencharm_top`, `fahexa750_ambencharm_top`. All six now have a
combined USD under `assets/meshes/usd/combined/` (see "Regenerating combined
USD assets" below) and can be opened with `examples/show_robot.py` -- see
`docs/ARCHITECTURE.md`'s asset table for which are wired into an actual
task vs. spawnable-but-not-yet-exercised.

Five multi-agent collaboration tasks (`environments/collaborative/`) are
registered and config-driven (`cfg/task/Aerial*.yaml`, each listing its
agents' presets under `aerial_manipulators:`) but are scaffolding only --
see `docs/ARCHITECTURE.md`'s asset table for which presets they use and
which of those can actually spawn today.

## Setup

```bash
pip install -e /home/benhua/OmniDrones   # aetos_multiam imports omni_drones directly
pip install -e .
```

Robot model binaries (USD + every MJCF bundle's mesh geometry) aren't
committed to this repo -- download robot assets from google_drive and unzip
into `aetos_multiam/assets/meshes` (see [`docs/ASSETS.md`](docs/ASSETS.md)
for the link and the exact layout).

**Every script below needs Isaac Sim on the interpreter's path.** Run them
through the `omnidrone` conda env with `ISAACSIM_PATH` exported first (its
own activation script silently no-ops without that env var, which a fresh
shell never has set) -- not Isaac Sim's own `python.sh`, which doesn't have
`aetos_multiam`/`omni_drones` installed:

```bash
export ISAACSIM_PATH=/home/benhua/isaacsim
conda activate omnidrone
```

## Running the demo task

```bash
python scripts/run_button_grasp.py headless=true
```

This has not been run inside this sandbox (booting `SimulationApp` needs a
live Isaac Sim session); run it yourself to verify before relying on it --
see `docs/ARCHITECTURE.md`'s "Known open risks" section.

## Viewing a robot preset

`examples/show_robot.py` opens an interactive Isaac Sim window for one
registered preset, or all of them side by side -- a quick way to check a
preset's USD exists and mounts correctly, independent of any task:

```bash
python examples/show_robot.py --robot iris650_soarm100_bottom
python examples/show_robot.py --robot all
python examples/show_robot.py --robot all --headless   # screenshot only, saved next to the script
```

Presets with no USD built yet are skipped with a message rather than
crashing the whole run.

## Teleoperating an arm with a real leader device

`examples/amteleop_soarm_leader.py` drives an aerial manipulator's
SO-ARM-family arm 1:1 from a physical lerobot SO101/SO-ARM100 leader arm
(read over USB-serial). Defaults to the `iris650_soarm100_bottom` preset;
any preset whose manipulator isn't SO-ARM-family is rejected at startup with
a clear error, since the leader-to-sim mapping is a direct per-joint mirror
onto SO-ARM joint names. The drone body is welded to the world (no flight
yet -- the script's own module docstring marks the extension points planned
for drone control and data recording).

```bash
python examples/amteleop_soarm_leader.py --port /dev/ttyACM0
# or, to exercise the sim side with no hardware connected:
python examples/amteleop_soarm_leader.py --fake
```

Full setup (serial permissions, first-run calibration) in
[`docs/TELEOP.md`](docs/TELEOP.md).

## Regenerating combined USD assets

Every file under `aetos_multiam/assets/meshes/usd/combined/` (except
`uaquad600_ambencharm_top.usd`, which needs bespoke re-mount surgery -- see
its own script) is produced by `scripts/export_robot_usd.py`, driven by a
YAML config per robot:

```bash
python scripts/export_robot_usd.py --config scripts/export_configs/iris650_soarm100.yaml
```

One config per output file lives under `scripts/export_configs/`; see
`scripts/export_robot_usd.py`'s own module docstring for the config format,
and [`docs/ASSETS.md`](docs/ASSETS.md) for when to re-run which one.

## Layout

```
aetos_multiam/
  robots/{drones,manipulators,grippers}/   # swappable component specs + registries
  robots/robot_base.py                     # AerialManipulatorBase: runtime composition
  robots/presets.py                        # named drone+arm+gripper combos (PRESETS)
  devices/                                 # physical teleop input devices (SOArmLeader + vendored Feetech driver)
  actuators/, controllers/                 # rotor thrust, arm/gripper/composite controllers
  scenes/, objects/                         # arenas and task objects (robosuite analogs)
  environments/                            # REGISTERED_ENVS + base classes
  environments/single/button_grasp.py      # the one spawnable single-AM task
  environments/collaborative/              # 5 config-driven multi-AM tasks (scaffolding)
  generation/                              # AetosTaskSpec -- schema for future demo generation
  utils/usd_compose.py                     # the drone x arm x gripper mounting mechanism
cfg/                                        # hydra configs (same shape as OmniDrones's cfg/)
scripts/
  run_button_grasp.py                      # smoke-test entry point
  export_robot_usd.py + export_configs/    # bake presets into standalone combined USDs
examples/
  show_robot.py                            # view one preset, or all of them
  amteleop_soarm_leader.py                 # real-leader-arm teleop (framework, see docs/TELEOP.md)
```
