# Aetos_MultiAM

A benchmark for **multi aerial-manipulator (AM) collaboration**: one robot
*type* -- an aerial manipulator (drone + optional arm + optional gripper) --
composed from swappable drone / manipulator / gripper models, built on Isaac
Sim's `omni.isaac.core` API.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how this is organized
and how it maps onto its reference projects (robosuite, dexmimicgen,
OmniDrones, AM-bench), including an honest accounting of what's a real,
working asset vs. registered scaffolding.

## What's new

1. **Six aerial manipulators, all of them flyable** -- from an under-actuated
   quadrotor to a fully-actuated hexarotor, with 2- to 6-DOF arms ending in a
   two-finger gripper, a three-finger gripper, or a five-finger dexterous hand.
   See [Embodiments](#embodiments).
2. **Keyboard teleoperation of the drone *and* the arm**, as two independently
   selected devices. See
   [Teleoperating an aerial manipulator](#teleoperating-an-aerial-manipulator).
3. **The fixes those six needed** before any of the above worked -- eight of
   them, each one a real defect in a vendored asset or in the spawn path. See
   [Fixes behind the six](#fixes-behind-the-six).

## Embodiments

Each clip is that robot flying `examples/show_am_action.py`'s scripted mission
at **3x speed**: climb to 1 m, forward 1 m, left 1 m, right 1 m, yaw 180 deg,
forward 1 m, sweep every drivable joint +-45 deg, land.

| preset | rotors | arm DOF | gripper DOF | mission (3x) |
|---|---|---|---|---|
| `amov950_airbot_bottom` | quad (4) | 6 | 2 (airbot_play) | <img src="resources/amov950_airbot_bottom.gif" width="240"> |
| `amov950_airbot_dexterous_bottom` | quad (4) | 5 | 12 (Inspire Hand) | <img src="resources/amov950_airbot_dexterous_bottom.gif" width="240"> |
| `hummingbird340_2dof_gripper` | quad (4) | 2 | 3 (three-finger) | <img src="resources/hummingbird340_2dof_gripper.gif" width="240"> |
| `iris650_soarm100` | quad (4) | 5 | 1 (SO-ARM jaw) | <img src="resources/iris650_soarm100.gif" width="240"> |
| `uaquad600_gripper` | quad (4) | 4 | 2 (two-finger) | <img src="resources/uaquad600_gripper.gif" width="240"> |
| `fahexa750_gripper` | **hexa (6)** | 4 | 2 (two-finger) | <img src="resources/fahexa750_gripper.gif" width="240"> |

`fahexa750_gripper` is a fixed-cant fully-actuated hexarotor (its 6x6 wrench
matrix has rank 6), but is flown here through the same 4-DOF mixer as the
other five -- its lateral authority is unused for now, see
[`docs/TELEOP_CONTROL_MODES.md`](docs/TELEOP_CONTROL_MODES.md).

A moving arm kicks the hovering drone. In the clips above it costs **under
0.7 deg of body tilt on every robot**, because each preset declares how much
of its arm's reaction mass to keep (`arm_dynamics_residual`, `arm_locked_joints`)
-- measured per robot in [`docs/ARM_DISTURBANCE.md`](docs/ARM_DISTURBANCE.md).

```bash
python examples/show_am_action.py --preset uaquad600_gripper
python examples/show_am_action.py --preset all     # all six in one scene, together
```

## Fixes behind the six

Each of these was a real defect, found by flying the robot rather than by
reading the asset; several passed every numeric flight check and were only
visible in the viewport.

| what broke | cause | fix |
|---|---|---|
| `hummingbird340` couldn't hover at all (187 deg yaw, 21 m off) | `ArticulationRootAPI` sat on the spawn root, so PhysX picked an arm link as the floating base | `_pin_articulation_root_to_drone()` moves the root onto `base_link` |
| `hummingbird340`'s arm never moved | the USD applies `DriveAPI` under the axis name (`X`/`Y`), not the `angular`/`linear` instance name the drive API reads | `normalize_joint_drives()` in `utils/usd_fixups.py` |
| every robot sagged ~0.5 m below its setpoint | the controller used the drone spec's authored mass, 16.5 % short of the composed articulation's real total | `initialize()` reads the real total from the articulation and overrides it |
| `uaquad600`'s arm hung under the airframe | `quad_scorpion.usd` mounts it below; this project's own MJCF bundle for the same robot mounts it on top | `joint_pose_overrides` on the preset, same transform the MJCF authors |
| `fahexa750`'s arm swept straight through the body | `hexa_scorpion.usd` authors `enabledSelfCollisions=False` (the other five author `True`) | `enable_self_collisions=True` on the preset |
| the Inspire hand's fingers went non-finite on the first commanded sweep | the asset authors them at `kp=0.05` next to arm joints at `kp=1e7` -- a 1e8 spread inside one articulation | `arm_write="teleport"` on that preset |
| two robots in one scene **segfaulted** inside `get_body_masses` | `initialize()`'s second `sim.reset()` is global -- it invalidates every *other* robot's physics view | `initialize(reset_rest_pose=False)` + `reattach()`, caller owns the one reset |
| asset fixups silently skipped separately-mounted arms | they ran before `mount_child`, so the arm didn't exist yet | moved to a second pass after the mount loop |

## Status

This is a first-pass scaffold. **One task is verified to run end-to-end:**
`ButtonGrasp` -- a 950mm AmovDrone + airbot_play arm flies up to a
wall-mounted button and presses it. `ShelfButtonGrasp` (see "Shelf grasp
scene" below) follows the exact same pattern and is written the same way,
but hasn't been booted in Isaac Sim yet -- run `run_shelf_button_grasp.py`
yourself to verify before relying on it. Every other drone/manipulator/
gripper registered under `aetos_multiam/robots/` is a data spec + registry
entry, not yet exercised by a task (see the asset table in
`docs/ARCHITECTURE.md`).

The six presets in [Embodiments](#embodiments) above
(`aetos_multiam.robots.PRESETS` / `robots/presets.py`, named
`<drone><manipulator>_<mount>`) each have a combined USD under
`assets/meshes/usd/combined/` (see "Regenerating combined USD assets" below),
open with `examples/show_robot.py`, and fly `examples/show_am_action.py`'s
mission -- see `docs/ARCHITECTURE.md`'s asset table for which are wired into
an actual task vs. spawnable-and-flyable-but-not-yet-exercised-by-one.

This project deliberately does **not** support picking an arbitrary
drone/manipulator/gripper combo at runtime -- these six are the fixed set.
The two `amov950_airbot*` presets spawn OmniDrones' own
`amov_drone_arm_1.usd` directly (`drones/amov_drone_arm.py`) rather than
mounting a separately-vendored copy of the arm at runtime: that vendored
copy's `arm_base` body massed 4.7x the real training asset's own
(0.777kg vs 0.166kg), and PhysX doesn't recompute its cached inertia
tensor from a bare mass-value fix, so a trained hover policy that's fine
on the real asset can still flip on the mismatched one. `amov950_airbot_dexterous_bottom`
deactivates that same file's stock 2-finger gripper (`link6` + its
`gripper_left`/`gripper_right` fingers) at spawn time and mounts the
Inspire Hand on `link5` instead (`manipulators/amov_arm_bare.py` +
`grippers/inspire_hand.py`). Both verified spawning in Isaac Sim
(`examples/show_robot.py --robot amov950_airbot_bottom` /
`--robot amov950_airbot_dexterous_bottom`, headless screenshot).

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
python examples/show_robot.py --robot iris650_soarm100
python examples/show_robot.py --robot all
python examples/show_robot.py --robot all --headless   # screenshot only, saved next to the script
```

Presets with no USD built yet are skipped with a message rather than
crashing the whole run.

## Teleoperating an arm with a real leader device

`examples/amteleop_soarm_leader.py` drives an aerial manipulator's
SO-ARM-family arm 1:1 from a physical lerobot SO101/SO-ARM100 leader arm
(read over USB-serial). Defaults to the `iris650_soarm100` preset;
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

Which preset can be flown and have its arm driven right now, with what
device, and how that compares to AM-Bench's EE-centric keyboard teleop:
[`docs/TELEOP_CONTROL_MODES.md`](docs/TELEOP_CONTROL_MODES.md).

`examples/show_am_action.py` flies any preset through one fixed mission
(climb, square-ish traverse, 180 deg turn, full-range arm sweep, land) as a
reproducible stand-in for the teleop devices that aren't wired up yet:

```bash
python examples/show_am_action.py --preset uaquad600_gripper
python examples/show_am_action.py --preset all     # all six side by side
```

A moving arm kicks the hovering drone. How much, why each robot needs
different settings, and the knobs that control it (`arm_dynamics_residual`,
`arm_locked_joints`) are in
[`docs/ARM_DISTURBANCE.md`](docs/ARM_DISTURBANCE.md).

## Shelf grasp scene

`ShelfButtonGrasp` (`cfg/task/ShelfButtonGrasp.yaml`) is a 4-tier chrome
shelf holding a gearwheel, a button, a power drill, and a tote box,
migrated from raids-am-simulator's `shelf_grasp.usd` scene
(`aetos_multiam/scenes/shelf_arena.py` + `aetos_multiam/objects/{gearwheel,
shelf_button,power_drill,tote_box}.py`; the object USD/mesh assets
themselves live under `aetos_multiam/objects/assets/`, not committed to git
-- see [`docs/ASSETS.md`](docs/ASSETS.md)). Defaults to the
`amov_airbot_gripper` robot (preset `amov950_airbot_bottom`):

```bash
python scripts/run_shelf_button_grasp.py task=ShelfButtonGrasp headless=true
```

Both the scene and the robot are explicit, swappable config: pick a
different scene via `task=<OtherTaskName>`, pick a different robot (any
preset in `robots/presets.py`) via a hydra override, no code change needed:

```bash
python scripts/run_shelf_button_grasp.py task=ShelfButtonGrasp \
    aerial_manipulators.0.preset=amov950_airbot_dexterous_bottom headless=true
```

This has not been run inside this sandbox either -- same caveat as
`run_button_grasp.py` above.

## Running the ShelfButtonGrasp task with real control sources

`examples/run_task_ShelfButtonGrasp.py` is the shelf scene actually *run* as
a task -- scene + something flying the drone + something moving the arm,
mirroring what raids-am-simulator could already do (drone via a controller,
arm+objects via a recorded trajectory), but with the drone/arm control
source each picked independently and explicitly:

```bash
python examples/run_task_ShelfButtonGrasp.py                      # drone=policy, arm=replay (defaults)
python examples/run_task_ShelfButtonGrasp.py --headless --max_steps 500
```

- `--drone-control {policy,device}` (default `policy`): `policy` loads a
  trained RL hover checkpoint (from the sibling OmniDrones repo's
  `task=HoverAmovDynaArmReplay` run) to keep the drone steady; `device` is
  an extension point (raises `NotImplementedError` -- no drone-control
  input device exists yet).
- `--arm-control {replay,policy,device}` (default `replay`): `replay`
  plays back a recorded raids-am-simulator episode (arm joints + gripper +
  all 4 object poses), migrated into
  `aetos_multiam/data/shelf_button_grasp/episodes/`; `policy`/`device` are
  extension points (raise `NotImplementedError` -- no manipulation policy
  or compatible leader device exists yet).

The robot is spawned directly from OmniDrones' own pre-baked
`amov_drone_arm_1.usd` (via OmniDrones' `AmovDrone` class), not
aetos_multiam's composable `amov950_airbot_bottom` preset -- see
"Why this script doesn't use aetos_multiam's preset system" below.

Six real fixes were needed to stop the drone from flipping -- kinematic
authored before the first `sim.reset()` under the GPU pipeline; every
internal joint in a prop deactivated before going kinematic; the arm's
joint position/velocity (not just its PD target) teleported to the
replay's starting frame before the first physics step; the recorded
trajectory (30 fps) remapped onto the sim's own control rate (~62 Hz)
with the arm held at frame 0 until the hover policy settles (both ported
from OmniDrones' own `scripts/run_shelf_grasp.py`); the arm's PD gains
matched to what the checkpoint actually trained with
(stiffness=500/damping=50); and the policy's raw action run through the
same hover-centered affine remap (`action*scale+offset`) `hover.py`'s own
`_pre_sim_step` applies before `apply_action` -- skipping it is a silent,
systematic thrust bias. See the script's own module docstring for the
full notes, including why the robot asset itself (not just these six
control-loop fixes) mattered: a separately-vendored copy of the arm used
by the preset system has one body (`arm_base`) at 4.7x the mass of the
training asset's own, and fixing the mass value alone didn't help because
PhysX's cached inertia tensor doesn't get recomputed from a bare
`set_body_masses()` call -- so this script just spawns the literal
training file instead.

**Verified in Isaac Sim (headless, GPU PhysX) against OmniDrones' own
`scripts/run_shelf_grasp.py` as a baseline**, same checkpoint, same
`ep000.npz` episode: both now complete the full 605-frame replay with zero
PhysX/Python errors and no flip, each recovering from a large mid-episode
disturbance spike (pos_err to ~1.2-1.8, `up_z` dipping to ~0.93-0.96 but
never approaching the flip threshold) rather than diverging. One cosmetic
gap remains: the button's 2 decorative sub-buttons are frozen wherever
they spawn (their joint is gone, nothing else drives them).

### Why this script doesn't use aetos_multiam's preset system

aetos_multiam's whole point is swappable drone x arm x gripper combos,
composed at runtime instead of needing a hand-baked USD per combination.
That's exactly the wrong tool for *this* script, whose goal is
reproducing one specific trained checkpoint's physical setup exactly --
the pre-baked merged USD gives that for free (same file, bit-for-bit) in
a way runtime composition of separately-vendored parts can't easily
guarantee (see the mass/inertia mismatch above). Use the preset system
(`AerialManipulatorBase` + `robots/presets.py`) when you want a different
drone/arm/gripper combo or don't need to match an existing checkpoint;
use the direct-USD approach this script takes when you do.

## Teleoperating an aerial manipulator

`examples/aerial_manipulator_control_demo.py` drives one robot's drone body
and arm from two independently-selected devices:

| flag | values | default |
|---|---|---|
| `--uav_control` | `none`, `keyboard` | `none` (holds its spawn setpoint) |
| `--arm_control` | `none`, `keyboard`, `soarm_leader` | `none` (holds its rest pose) |
| `--arm_dynamics` | `frozen`, `coupled` | `frozen` (arm doesn't push back) |
| `--arm-write` | `auto`, `drive`, `teleport` | `auto` (the preset decides) |

Keys -- drone: `WASD` translate, `E`/`Q` up/down, `Left`/`Right` yaw. Arm:
`1`-`9` select a joint, `[`/`]` move it, `G`/`H` close/open every gripper DOF,
`Z` home the arm. The arm bindings are generic over whatever DOFs the preset
actually has, so the same keys drive a 2-DOF arm, a 6-DOF arm, or a
12-DOF dexterous hand -- `aetos_multiam/devices/keyboard_arm_control.py`.

`--arm_dynamics frozen` keeps the arm's static mass and inertia -- lumped onto
`base_link` -- but leaves the links themselves with only a small residual
fraction of theirs, so swinging the arm transmits almost no reaction wrench
into the drone body and a controller tuned for a fixed airframe stays stable.
Same "simplify the plant first, learn the real coupled dynamics later" split
humanoid pipelines use between retargeting and the learned controller.
`--arm_dynamics coupled` restores the real multibody physics.
`aetos_multiam/utils/arm_dynamics.py` spells out exactly what is and isn't
preserved, and why the residual is a fraction rather than a flat epsilon.

What robot gets spawned comes from `--task <SceneName>` (the scene's own
`cfg/task/<name>.yaml` declares it -- `ShelfBucketGrasp` → `iris650_soarm100`),
`--preset <name>` (that preset on a bare ground plane), or `--drone-model`
(a bare OmniDrones drone with no arm, default `Hummingbird`).

```bash
# scene's own robot, nothing driven -- just holds station
python examples/aerial_manipulator_control_demo.py --task ShelfBucketGrasp --mode dynamic

# fly it with WASD / E,Q / arrows
python examples/aerial_manipulator_control_demo.py --task ShelfBucketGrasp --mode dynamic \
    --uav_control keyboard

# ...and drive the arm from a real SO-ARM leader at the same time
python examples/aerial_manipulator_control_demo.py --task ShelfBucketGrasp --mode dynamic \
    --uav_control keyboard --arm_control soarm_leader --arm-port /dev/ttyACM0

# no hardware? sine-sweep the arm instead
python examples/aerial_manipulator_control_demo.py --task ShelfBucketGrasp --mode dynamic \
    --arm_control soarm_leader --arm-fake

python examples/aerial_manipulator_control_demo.py --mode dynamic   # bare drone, no scene
```

`aetos_multiam/devices/keyboard_uav_control.py`'s `KeyboardUAVControl` is
just the input device (polls key state each frame, no event callbacks --
same style as `SOArmLeader`); `--mode` picks which flight model realizes its
intent:

- `--mode kinematic`: ignores the drone's rotor dynamics entirely -- the
  keyboard intent integrates into a target pose that's teleported straight
  onto the body every step. "Fly a fixed point in space," no body tilt ever.
- `--mode dynamic`: the same target pose is instead chased by OmniDrones'
  own `LeePositionController` under real gravity, via real per-rotor thrust
  (`drone.apply_action`). Verified directly (forced forward+ascend command,
  headless): the body pitches forward (quaternion x-component peaks around
  -0.12) while accelerating toward a moving target, then levels back out
  once it catches up to the target's velocity -- the same forward-lean a
  real quadrotor shows, a side effect of the controller's own geometric
  control law, not hand-added.

## Regenerating combined USD assets

Every file under `aetos_multiam/assets/meshes/usd/combined/` (except
`uaquad600_gripper.usd`, which needs bespoke re-mount surgery -- see
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
  devices/                                 # teleop input devices (SOArmLeader + vendored Feetech driver,
                                            #   keyboard_uav_control.py -- polled WASD/E/Q/arrow UAV flight stick,
                                            #   keyboard_arm_control.py -- per-joint arm/gripper keys)
  actuators/, controllers/                 # rotor thrust, arm/gripper/composite controllers
  scenes/, objects/                         # arenas and task objects (robosuite analogs)
  objects/assets/                          # object USD/mesh binaries (gitignored, see docs/ASSETS.md)
  data/shelf_button_grasp/episodes/        # recorded arm+object replay episodes (tracked, small)
  environments/                            # REGISTERED_ENVS + base classes
  environments/single/button_grasp.py      # the one spawnable single-AM task
  environments/single/shelf_button_grasp.py # shelf + 4 props, migrated from raids-am-simulator
  environments/collaborative/              # 5 config-driven multi-AM tasks (scaffolding)
  generation/                              # AetosTaskSpec -- schema for future demo generation
  utils/usd_compose.py                     # the drone x arm x gripper mounting mechanism
cfg/                                        # hydra configs (same shape as OmniDrones's cfg/)
scripts/
  run_button_grasp.py                      # smoke-test entry point
  run_shelf_button_grasp.py                # smoke-test entry point for the shelf scene
  export_robot_usd.py + export_configs/    # bake presets into standalone combined USDs
examples/
  show_robot.py                            # view one preset, or all of them
  amteleop_soarm_leader.py                 # real-leader-arm teleop (framework, see docs/TELEOP.md)
  run_task_ShelfButtonGrasp.py             # shelf scene + pluggable drone/arm control sources
  aerial_manipulator_control_demo.py             # drone+arm teleop: --uav_control / --arm_control, see above
```
