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

Five named aerial-manipulator presets exist (`aetos_multiam.robots.PRESETS`
/ `robots/presets.py`), one per combo asset bundle under `assets/mjcf/`,
named `<drone><manipulator>_<mount>`: `amov950_airbot_bottom` (spawnable),
`hummingbird340_dynamixel2r_bottom`, `iris650_soarm100_bottom`,
`uaquad600_ambencharm_top`, `fahexa750_ambencharm_top` (the latter four are
MJCF source only, pending USD conversion -- see `docs/ARCHITECTURE.md`).

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

## Running the demo task

```bash
/home/benhua/isaacsim/python.sh scripts/run_button_grasp.py headless=true
```

This has not been run inside this sandbox (booting `SimulationApp` needs a
live Isaac Sim session); run it yourself to verify before relying on it --
see `docs/ARCHITECTURE.md`'s "Known open risks" section.

## Layout

```
aetos_multiam/
  robots/{drones,manipulators,grippers}/   # swappable component specs + registries
  robots/robot_base.py                     # AerialManipulatorBase: runtime composition
  actuators/, controllers/                 # rotor thrust, arm/gripper/composite controllers
  scenes/, objects/                         # arenas and task objects (robosuite analogs)
  environments/                            # REGISTERED_ENVS + base classes
  environments/single/button_grasp.py      # the one spawnable single-AM task
  environments/collaborative/              # 5 config-driven multi-AM tasks (scaffolding)
  generation/                              # AetosTaskSpec -- schema for future demo generation
  utils/usd_compose.py                     # the drone x arm x gripper mounting mechanism
cfg/                                        # hydra configs (same shape as OmniDrones's cfg/)
scripts/run_button_grasp.py                # smoke-test entry point
```
