# Aetos_MultiAM architecture

A benchmark for **multi aerial-manipulator (AM) collaboration**: one robot
*type* (drone + optional arm + optional gripper), many swappable
configurations, built on Isaac Sim's raw `omni.isaac.core` API (no IsaacLab).

## Where each idea came from

| Reference project | What was borrowed | Where it landed |
|---|---|---|
| **robosuite** | `models/{robots,grippers,bases,objects,arenas,tasks}` split; a runtime `Robot` wrapper merging model+gripper+controller; `environments/base.py`'s `REGISTERED_ENVS`/`make()` | `robots/{drones,manipulators,grippers}/`, `robots/robot_base.py:AerialManipulatorBase`, `environments/__init__.py` |
| **dexmimicgen** | flat `environments/<family>/<task>.py`, one file per concrete task; import-side-effect registration | `environments/single/button_grasp.py`; `environments/single/__init__.py` imported from `environments/__init__.py` |
| **NVIDIA mimicgen** (not dexmimicgen's own replay scripts) | `MG_TaskSpec`: ordered subtasks + termination signals, for future automated demo generation | `generation/task_spec.py:AetosTaskSpec` (schema only, no generator yet) |
| **OmniDrones** (this repo) | `RobotBase` registry, `MultirotorBase`/`RotorGroup` thrust physics, `ControllerBase`, `IsaacEnv`'s torchrl/`GridCloner` lifecycle | imported directly (see "Dependency on OmniDrones" below), plus `environments/base_env.py:AetosEnv` wraps `IsaacEnv` |
| **AM-bench** (`/home/benhua/ambench`) | `RobotSpecCfg`/`GripperSpecCfg` shape (joints + open/closed calibration); the `press_button` task concept; its `push_button.usd` asset | `robots/manipulators/manipulator_base.py`, `robots/grippers/gripper_base.py`, `environments/single/button_grasp.py`, `objects/button.py` |

AM-bench itself is built on IsaacLab, which isn't installed in this
environment -- its *software shape* was ported, not its IsaacLab-specific
code.

## The composable-mounting mechanism

robosuite merges robot+gripper by splicing MJCF XML trees before `MjSim` is
built. There's no XML tree in Isaac Sim -- `utils/usd_compose.py:mount_child`
does the equivalent: reference the child's USD as a **sibling** of the
parent body's prim (not nested inside it -- see below), strip the child's
own `ArticulationRootAPI` (if present -- it was very likely authored to also
stand alone), then weld it to the parent with a PhysX fixed joint so the
whole thing simulates as **one** articulation instead of two
loosely-coupled ones.

`AerialManipulatorBase.spawn()` (`robots/robot_base.py`) calls this twice:
drone `base_link` -> manipulator root, then manipulator `eef_body_name` ->
gripper root (skipped when the gripper is already bundled into the
manipulator's own USD, e.g. `airbot_play`).

**Found and fixed by actually running it (2026-10-29):** the first version
of `mount_child` referenced the child *nested inside* the parent body prim
(`{parent_body_path}/{child_mount_name}`) -- PhysX rejects any RigidBodyAPI
prim nested inside another RigidBodyAPI prim ("xformstack reset ... missing
when child of rigid body in hierarchy"), and every joint inside the nested
child then failed to resolve at all ("CreateJoint - no bodies defined"),
silently producing an articulation with the manipulator's joints simply
absent -- confirmed live via `scripts/run_button_grasp.py`, which crashed
downstream with `arm_joint_indices` lookup failing because `joint1..joint6`
genuinely weren't in the merged articulation's `dof_names`. This is the
exact same nesting bug the Fin-Ray finger work (see the finger section
below) ran into and fixed independently -- the fix here is the same:
reference the child as a sibling, computing its absolute world pose by hand
(`UsdGeom.Xformable(...).ComputeLocalToWorldTransform()` on the parent body,
composed with the requested relative offset) since siblings don't inherit
the parent's transform the way USD parent/child prims do.
`omni.physx.scripts.utils.createJoint()` then derives the weld's local
joint frames from the two prims' actual world poses regardless of where
either lives in the hierarchy, so this doesn't need any further change.
**Live-verified after the fix**: `scripts/run_collab_smoke.py task=AerialTransport`
and `task=AerialAssemblyTriple` both spawn their `amov950_airbot_bottom`
agents (2 and 3 of them respectively) as correctly-merged 12-DOF
articulations (`rotor_joint_1..4` + `joint1..6` + `endleft`/`endright`) and
run `reset()` + 20 steps of `step()` with zero actions with no errors --
the first time *any* multi-body composition in this project has actually
been run, not just authored.

## Registry / "robot and controller type" config

One `AerialManipulatorCfg` (`drone_type`, `manipulator_type`, `gripper_type`,
`controller_type`) selects a combination, analogous to robosuite's
`robots=[...]`, `controller_configs=...` env kwargs collapsed into one
dataclass since there's only one robot *type* here. `robots/__init__.py`'s
`make_aerial_manipulator(cfg)` is the one-call factory
(`create_robot`+`gripper_factory`+`robot_base_factory` rolled into one).

`environments/__init__.py`'s `REGISTERED_ENVS` + `make(env_name, cfg,
headless)` is the literal answer to "通过 environment 模块注册环境类" --
every task gets listed there. Implemented via `AetosEnv.__init_subclass__`
(not a real metaclass like robosuite's `EnvMeta`, to avoid clashing with
torchrl `EnvBase`'s own metaclass).

## Dependency on OmniDrones

Like dexmimicgen imports robosuite rather than reimplementing MJCF merging,
`aetos_multiam` imports `omni_drones` for pieces that are generic simulation
infrastructure, not benchmark logic: `RotorGroup` (actuators),
`ControllerBase`/`LeePositionController` (controllers), `IsaacEnv` (the
torchrl `EnvBase` + `GridCloner`/`SimulationContext` bootstrap --
`environments/base_env.py:AetosEnv` subclasses it directly), and the
`Amov950Drone`/`Hummingbird`/`Firefly` drone physics classes themselves.
`aetos_multiam`'s own code is the composition layer on top: drone+arm+
gripper mounting, the registry/factory plumbing, and the task environments
-- none of which exist in OmniDrones today.

If this benchmark should instead be fully standalone (vendored copies, no
import of `omni_drones`), that only touches `actuators/rotor_group.py`,
`controllers/drone/lee_position_controller.py`, and
`environments/base_env.py` -- nothing else in this layout changes.

## Multi-agent collaboration tasks (`environments/collaborative/`)

Five tasks, each a heterogeneous or homogeneous team of aerial manipulators
declared entirely by `cfg/task/Aerial*.yaml`'s `aerial_manipulators:` list
(one entry per agent, by preset name) -- adding a new team composition is a
config change, not a code change. All five share
`CollaborativeAerialEnv` (`environments/collaborative/collaborative_base.py`)
for the generic plumbing; see that file's own docstring for why it diverges
from this repo's `Formation`/`Rearrange` pattern (one shared stacked
`"agents"` tensor doesn't work once agents can have different arm-joint
counts) in favor of one top-level spec key per agent (`f"agent_{i}"`).

| Task | Agents (presets) | Spawnable today? |
|---|---|---|
| `AerialHandover` | `hummingbird340_dynamixel2r_bottom`, `uaquad600_ambencharm_top` | No -- both MJCF-only |
| `AerialAssembly` | `uaquad600_ambencharm_top`, `amov950_airbot_bottom` | No -- one MJCF-only |
| `AerialTransport` | `amov950_airbot_bottom` x2 | **Yes** |
| `AerialCabinetGrasp` | `fahexa750_ambencharm_top` x2, `iris650_soarm100_bottom` | No -- all MJCF-only |
| `AerialAssemblyTriple` | `amov950_airbot_bottom` x3 | **Yes** |

None of the five have a real task-object asset yet (a handover target,
assembly fixture, shared payload, or cabinet) -- `CollaborativeAerialEnv`'s
reward/success is a generic "converge near a per-agent goal point"
placeholder, not real task logic. `AerialTransport`/`AerialAssemblyTriple`
are the two worth actually smoke-testing first, since they're the only ones
built entirely from the one preset (`amov950_airbot_bottom`) that can spawn.

## Asset status (what's real vs. scaffolding)

Five named presets (`robots/presets.py:PRESETS`) cover the five
drone+arm(+gripper) bodies this project has source material for. Only the
first is actually spawnable today; the other four each have **real,
precisely captured** source data (exact joint names, mount transforms,
mass, and where available rotor allocation constants, all cited to a
specific file), but each is blocked on a *different* remaining gap --
not uniformly "needs MJCF/USD conversion" any more:

- `uaquad600_ambencharm_top`/`fahexa750_ambencharm_top`: USD already exists
  (AM-bench is itself Isaac-Sim-based) -- only missing a `RotorGroup`-compatible
  KF/KM yaml (a calibration choice, not a lookup).
- `hummingbird340_dynamixel2r_bottom`: the dynamixel 2R arm itself is fine
  (plain MJCF, importable); the three-finger gripper's soft `flexcomp`
  fingers have no PhysX equivalent and need a rigid approximation (see
  "Fin Ray finger reproduction" below).
- `iris650_soarm100_bottom`: rotor physics is fully specified, but no
  *scaled* USD body exists for the 650mm Iris, and SO-ARM100 has no local
  or found Isaac Sim USD either (community Isaac Lab ports exist upstream,
  not fetched into this repo -- see that section below).

| Preset | Drone | Manipulator | Gripper | Status |
|---|---|---|---|---|
| `amov950_airbot_bottom` | `amov_950` | `airbot_play` | bundled | **Spawnable.** Used by `ButtonGrasp`. |
| `hummingbird340_dynamixel2r_bottom` | `omnidrones_340` | `dynamixel_2r` | `three_finger_softgripper` | Real MJCF source (`assets/mjcf/hummingbird_2r_three_finger_50pct/`), validated hover/joint/gripper tests in MuJoCo. Gripper fingers are MuJoCo `flexcomp` soft bodies -- no PhysX equivalent, needs a rigid approximation. |
| `iris650_soarm100_bottom` | `iris_650` | `so_arm100` | bundled (`Jaw` joint) | Real MJCF source (`assets/mjcf/iris_650_so_arm100_final/`); rotor physics is fully specified (`assets/usd/iris_650.yaml`, exact KF/KM/max-speed from the source bundle) but no scaled USD body exists yet. |
| `uaquad600_ambencharm_top` | `ambench_600` | `ambench_quad_arm` | bundled | **Real Isaac Sim USD** (`quad_scorpion.usd`, AM-bench is itself Isaac-Sim-based -- see "AM-bench's own USD assets" below). `ambench_quad_arm` is pre-baked into that same USD (`bundled_into_drone=True`), not a separate mount. Only remaining gap: AM-bench's own rotor model has no KF/KM to port -- that pair would need to be chosen, not read off. |
| `fahexa750_ambencharm_top` | `ambench_750` | `ambench_hexa_arm` | bundled | Same status as the quad preset (`hexa_scorpion.usd`); tilt-rotor hexacopter. |

Naming convention: `<drone size+family><manipulator type>_<mount position>`.
Mount position (`top`/`bottom`) is cited from each source bundle's own docs
where available (`amov950_airbot_bottom` is the one exception -- inferred
from this repo's own project memory on the arm-base joint force sign, not a
bundle doc, since that combo predates the downloaded bundles).

| Component | Status |
|---|---|
| `amov_950` drone | **Real**, used by `ButtonGrasp`. Bare 950mm AmovDrone body, `omni_drones/robots/assets/usd/amov_drone.usd`. |
| `airbot_play` manipulator+gripper | **Real**, used by `ButtonGrasp`. Self-contained USD at `/home/benhua/DISCOVERSE/airbot_play_usd/my_airbot_play_v3_gripper/`. Joint/body names (`arm_base`, `link1..link6`, `joint1..joint6`, `endleft`/`endright`) are sourced from the sibling MJCF, **not** a direct USD read -- no working USD Python bindings were available in the sandbox this was built in. **Verify these against the live stage before relying on them.** |
| `Button` task object | **Real**, AM-bench's own `push_button.usd`, reused as-is (plain USD, simulator-agnostic). `pressed_threshold = 0.004` m is AM-bench's own calibrated value. |
| `omnidrones_430` (Firefly) | Real asset + physics, registered, **not exercised by any task yet**. |
| `dynamixel_2r` | Real MJCF source now (`assets/mjcf/hummingbird_2r_three_finger_50pct/robot.xml`): mount `arm_carrier` -> `base_2` -> `first_segment` (joint `R1`, +/-1.9199 rad) -> `second_segment` (joint `R2`, +/-2.6180 rad) -> `tool_adapter`. These are the *50%-scaled* combo's numbers, not the full-size arm's. Still no USD. |
| `so_arm100` | Real MJCF source (`assets/mjcf/iris_650_so_arm100_final/robot.xml`), full original size: `Base` -> `Rotation`/`Pitch`/`Elbow`/`Wrist_Pitch`/`Wrist_Roll` -> gripper joint `Jaw`. Servo params (kp=50, dampratio=1, armature=0.1, frictionloss=0.1, +/-3.5 N*m) preserved from source. This is what "lerobot standard openarm" turned out to actually be -- lerobot's SO-ARM100, not a separately-named "OpenArm" model. Still no USD. |
| `ambench_quad_arm`, `ambench_hexa_arm` | **Real**, pre-baked into `ambench_600`/`ambench_750`'s own Isaac Sim USD (`bundled_into_drone=True` -- see below) -- not a separate asset to extract. These are **two different arms** despite both being "4 rotational + 2 gripper slide": quad uses yaw/pitch/pitch/pitch axes, hexa uses pitch/pitch/pitch/roll. |

### AM-bench's own USD assets (corrected from an earlier pass)

AM-bench is itself built on IsaacLab/Isaac Sim, so its asset repo
(`/home/benhua/ambench/source/ambench/ambench/assets/robots/`) already has
native Isaac Sim USD files -- `quad_scorpion.usd` (22MB, self-contained)
and `hexa_scorpion.usd` (a thin layer referencing sibling files in that
same directory's `configuration/` subfolder; resolve it from its own
location, don't copy just that one small file elsewhere). An earlier pass
through this project mistakenly treated these as needing the MJCF→USD
conversion path like the OmniDrones-side combos -- they don't. AM-bench
authored drone+arm+gripper as **one combined articulation** per robot, so
`ambench_quad_arm`/`ambench_hexa_arm` set `bundled_into_drone=True`
(`AerialManipulatorBase.spawn()` skips `mount_child` for them entirely;
`DroneSpec`/`ManipulatorSpec.check_asset()` were both updated so this
doesn't silently skip checking the one thing that actually is still
missing: a `RotorGroup`-compatible KF/KM yaml). The MJCF viewer exports
under `assets/mjcf/ambench_*` are kept as a secondary, cross-checked
reference for joint names/transforms, not the primary source any more.
| `inspire_hand` | **No asset found locally.** Registered as a data spec with a placeholder path; raises `AssetNotAvailableError` if actually spawned. |
| `three_finger_softgripper` | Source MJCF uses MuJoCo `flexcomp` soft fingers, which have **no PhysX/Isaac Sim equivalent** -- needs a rigid-approximation USD authored first. Kinematics (one driven hinge + two mimic joints) are already captured correctly from the source `<equality>` block, confirmed present in both the original source and the new 50%-scale combo. |

### Searched for ready-made Isaac Sim assets (SO-ARM100, dynamixel 2R)

Neither exists locally (checked the filesystem directly -- no `.usd`/`.urdf`
for either beyond what's already in this project's own `assets/mjcf/`).
Web search turned up:

- **SO-ARM100/101**: several community Isaac Lab external-project repos
  convert it from URDF to USD -- e.g. `Coded190/isaac_so_arm101`,
  `MuammerBay/isaac_so_arm101`, `smalleha/isaac_so_arm101` on GitHub, and a
  HuggingFace dataset repo (`johnnynunez/so101_assets`) with actual `.usd`
  files uploaded. These are unofficial community conversions, and mostly
  target **SO-ARM101** (a later hardware revision), not SO-ARM100 exactly
  -- close enough in practice (same lineage, similar joint layout) but not
  identical. None of these have been fetched into this repo; doing so
  needs the user's go-ahead (downloading a file requires explicit
  permission) and a check of whether SO-ARM101's joint layout actually
  matches what `manipulators/so_arm100.py` currently assumes from the MJCF
  bundle.
- **dynamixel 2R**: no dedicated Isaac Sim USD found anywhere, official or
  community. It's confirmed to originate from `google-deepmind/mujoco_menagerie`
  (matches this project's own `hummingbird_2r_three_finger_50pct` bundle's
  README, which cites the same source) -- the expected path is converting
  it directly via Isaac Sim's own `omni.importer.mjcf` (confirmed installed,
  see below), either from this project's existing scaled-down combo bundle
  or by pulling the full-size model from upstream `mujoco_menagerie` (this
  repo's local menagerie checkout only has `universal_robots_ur10e`, not
  `dynamixel_2r` -- it would need fetching from the upstream GitHub repo).

## Fin Ray finger reproduction (finger_1 done in both MuJoCo and Isaac Sim; finger_2/3 + gripper integration pending)

The three-finger gripper's soft fingers (MuJoCo `flexcomp`, no PhysX
equivalent -- see the asset table above) need a rigid-body approximation
before they can exist in Isaac Sim at all. First step, done: reproduce
*one* finger (`finger_1`) as a 3-rigid-segment pseudo-rigid-body
approximation, at `assets/mjcf/three_finger_fin_ray_rigid/finger_rigid.xml`
(+ `scene.xml`, `DERIVATION.md`). Every dimension (segment length/width/
mass, joint stiffness) traces back to the original flexcomp's point cloud
and material properties via an explicit, documented calculation -- not
eyeballed. **Actually validated**, not just written: loaded and stepped in
real MuJoCo (`mujoco==3.13.0`, available locally in the `airbotsim` conda
env) -- compiles, stays numerically stable under gravity, and curls
plausibly under a test tip load (see that directory's preview renders).

Second step, also done: the SAME structure authored directly as a PhysX
articulation in Isaac Sim (not an `omni.importer.mjcf` conversion), at
`assets/usd/three_finger_fin_ray_rigid/build_finger_1_rigid.py`, exporting
a real `finger_1_rigid.usd` next to it. Genuinely run against the local
Isaac Sim 4.1.0 install (via the `omnidrone` conda env's `ISAACSIM_PATH`),
not just authored and assumed correct -- settles under gravity to
`[0.86, 1.21, 1.56]` deg (base->tip) and curls mostly at the tip under a
0.3N lateral test load, matching the MuJoCo validation's qualitative
result, and the exported `.usd` was round-trip tested (re-referenced into
a fresh stage, re-simulated, same settled angles). Found and documented 3
real USD-Physics authoring pitfalls along the way (nested RigidBodyAPI
prims rejected, kinematic bodies rejected inside an articulation, and
critically: `LocalRot0`/`LocalRot1` silently produce a 0-DOF rigid weld if
left unauthored, with zero error) -- full writeup in that MJCF directory's
`DERIVATION.md` section 7, since it's the same derivation these numbers
come from either way.

Both representations also have a rendered video of the deformation
actually happening (not keyframed): `render_bend_video.py` (MuJoCo) and
`render_bend_video_isaac.py` (Isaac Sim, via `get_active_viewport()` +
`capture_viewport_to_file()` -- `omni.replicator.core`'s render-product
path returned a flat gray image for this scene with no error, so this
project uses the viewport-capture path instead; see DERIVATION.md section
9 for what else didn't work before this did).

Not yet done: the same treatment for `finger_2`/`finger_3` in either
format, reassembling all three into the full gripper (reusing the
original's `finger_2_synchronized`/`finger_3_synchronized` equality
constraints, which only ever synced the *actuated* base hinges, not the
flex joints), and mounting this under a real gripper housing instead of a
standalone world anchor.

## Inspire Hand (two real variants, extracted from larger public scenes)

No official standalone Inspire Hand USD exists anywhere found (Isaac Sim's
own "OpenUSD tuning tutorials" use an Inspire Hand as their teaching
example but ship only a URDF, not a USD, bundled with a newer Isaac Sim
installer than the one in this environment). Found two public USD scenes
that each embed a real Inspire Hand inside a bigger robot instead
(`github.com/konu-droid/robots_usd`, downloaded with the user's go-ahead),
and extracted each hand's own body/joint/mesh subtree into a standalone
asset at `assets/usd/inspire_hand/`:
`inspire_hand_g1dfq_{left,right}.usd` (from Unitree G1's bimanual hands,
12 fully-independent revolute joints/hand) and `inspire_hand_kuka_right.usd`
(from a Kuka-arm demo scene, a different, more underactuated-looking
Inspire Hand variant -- 12 revolute + 5 fixed-coupled joints). Registered
as three `GripperSpec` entries in `robots/grippers/inspire_hand.py`
(replacing an earlier placeholder with a guessed joint-naming scheme).

The extraction (`assets/usd/inspire_hand/extract_inspire_hands.py`) hit
two non-obvious USD pitfalls worth remembering for any future "cut a
subtree out of someone else's big USD file" job:
1. The G1 source's visuals/collisions aren't inline meshes -- each
   references a same-layer top-level prim named like
   `/Flattened_Prototype_26` (an internal mesh-dedup scheme, unrelated to
   USD's own native scenegraph instancing -- `stage.GetPrototypes()`
   reports a completely different `/__Prototype_N` set, and
   `Stage.Flatten()` does NOT resolve these dangling references either,
   both confirmed directly). Fix: discover every `Flattened_Prototype_N` a
   body subtree references (walking the **raw layer**, not the composed
   stage -- see point 2) and copy those prims in too.
2. Those same prototype prims are authored with `specifier = over`, which
   `prim.GetChildren()` and `Usd.PrimRange`'s **default** predicate both
   silently skip the children of entirely (confirmed: empty either way,
   while `GetAllChildren()`/`Usd.PrimAllPrimsPredicate` find them fine).
   Any traversal over copied/extracted content needs the all-prims
   predicate explicitly, or it'll silently miss real prims -- in this case
   it meant `Sdf.CopySpec()`'s un-remapped relationship targets (joint
   body0/body1 rels, `material:binding`) went unfixed inside every
   prototype, each one still pointing at the *source* file's container
   path instead of the new one, Isaac Sim just logs "Unresolved
   reference"/"refers to a path outside the scope" and silently falls back
   to a default material rather than erroring loudly.

Live-verified after both fixes: all three extracted files load as clean,
correctly-named 12-DOF articulations with real (not placeholder) textured
geometry. Not yet done: deriving a real mount pose for either variant onto
an arm's `eef_body_name` (both currently have no `mount_translation`
decided), and `GripperSpec.format_action`'s single-scalar open/closed
abstraction undersells what's actually a 12-DOF dexterous hand -- fine for
registering the asset, not for real per-finger grasp control.

**Follow-up (same day):** the two G1 variants' `hand_base` carried TWO
separate meshes in their visuals/collisions prototypes, not one --
`{side}_hand_base_link` (the real Inspire Hand base, ~88k points) and
`{root_prefix}` i.e. `left_wrist_yaw_link`/`right_wrist_yaw_link` (the
Unitree wrist servo's own adapter-flange geometry, ~20k points, a much
smaller/compact bbox -- G1-specific mounting hardware, not part of the
hand). User asked for the flange removed. Since it's a genuinely separate
`Mesh` prim under the same referenced prototype (not baked into one
merged mesh), a plain `Stage.RemovePrim()` on it was enough -- doesn't
touch `hand_base`'s own mass/CoM/diagonal-inertia, which are explicit
authored attributes, not derived from collider geometry, confirmed by
checking before removing. `extract_inspire_hands.py` now does this
removal automatically as its last step before saving (so a re-run stays
consistent), live-reverified after: both hands still 12-DOF, zero
unresolved references. The Kuka variant's `hand_base` has no analogous
issue (it's that model's own real palm link, not borrowed wrist hardware
from a different robot) -- left untouched.

## Known open risks (flagged, not resolved, in this pass)

- ~~USD prim/joint names for `airbot_play` were inferred, not read from the
  live stage~~ -- **resolved 2026-10-29**: confirmed live, the merged
  articulation really does expose `joint1..joint6`/`endleft`/`endright`.
- ~~`mount_child`'s `ArticulationRootAPI` stripping hasn't been
  runtime-verified~~ -- **resolved 2026-10-29**, and a real bug was found
  and fixed in the process (nested-RigidBody hierarchy rejected by PhysX --
  see "The composable-mounting mechanism" above). `AerialTransport` and
  `AerialAssemblyTriple` (both built on `amov950_airbot_bottom` x2/x3) are
  now confirmed spawnable and steppable via `scripts/run_collab_smoke.py`.
  `ButtonGrasp` gets past the mount now too, but its own
  `_setup_references()` hits a separate, unrelated bug: `RuntimeError: CUDA
  error: device-side assert triggered` at `dof_limits[self.drone.arm_joint_indices]`
  in `environments/single/button_grasp.py` -- not yet root-caused (the
  index itself looks in-range against the 12-DOF merged articulation;
  needs `CUDA_LAUNCH_BLOCKING=1` for a synchronous trace). This code path
  is specific to `ButtonGrasp` -- `CollaborativeAerialEnv` doesn't call
  `get_dof_limits()` anywhere, which is why the two tasks above aren't
  affected by it.
- **`AerialAssembly`/`AerialCabinetGrasp` still can't spawn** -- not a
  mounting-mechanism problem, just missing assets: `ambench_600`/
  `ambench_750` (the quad/hexa AM-bench drones) have real USD bodies but no
  `RotorGroup`-compatible KF/KM yaml (AM-bench's own rotor model has no
  such pair to port, it's a calibration choice, not a missing lookup), and
  `iris_650` has real rotor-physics params but no scaled USD body (only
  stock, unscaled `iris.usd` exists upstream).
- Everything in the asset table marked "not exercised" or "no asset found"
  is exactly that -- registered scaffolding, not tested code.
- **`iris_650`'s rotor ordering**: the source bundle's rotor positions
  (used for `rotor_angles` in `assets/usd/iris_650.yaml`) don't match a
  naive uniform scaling of stock `iris.yaml`'s own `rotor_angles` -- see
  that yaml's own comment. Could be CG-rebalancing after adding the arm, or
  a rotor-index mismatch between the two sources. Check against the live
  USD's actual `rotor_link_N` prims before trusting `directions`/signs.
- **`so_arm100`'s mount orientation** (a 180 deg flip composed with a
  further -90 deg body-frame yaw, per its source bundle's `README.md`) is
  described but not reduced to a verified quaternion in
  `manipulators/so_arm100.py` -- compute and check it against the live
  mounted pose rather than trusting an untested derivation.
- The `assets/mjcf/` bundles are themselves explicit about their own scope:
  each is a MuJoCo *kinematic viewer* export with "原Isaac Sim控制器、空气
  动力和任务环境未移植" (original Isaac Sim controller/aerodynamics/task
  environment not ported) -- they're a geometry/joint/mass source, not a
  physics-validated drop-in.
