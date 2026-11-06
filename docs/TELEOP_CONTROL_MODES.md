# Teleoperating the fleet: what drives the drone, what drives the arm

One table to answer "can I fly this robot and move its arm right now, and
with what". Everything below is measured on the presets in
`aetos_multiam/robots/presets.py`, through
`examples/aerial_manipulator_control_demo.py`.

## Status per preset

| preset | drone | arm DOFs | arm teleop today | arm teleop planned | residual |
|---|---|---|---|---|---|
| `amov950_airbot_bottom` | keyboard ✅ | 6 + 2 gripper | keyboard ✅ | exoskeleton | 0.001 |
| `amov950_airbot_dexterous_bottom` | keyboard ✅ | 5 + 12 Inspire hand | keyboard ✅ | Apple Vision Pro | 0.001 |
| `hummingbird340_2dof_gripper` | keyboard ✅ | 2 + 3 fingers | keyboard ✅ | keyboard (final) | 0.1 |
| `iris650_soarm100` | keyboard ✅ | 5 + gripper, `shoulder_pan` locked | keyboard ✅, SO-ARM leader ✅ | exoskeleton | 0.1 |
| `uaquad600_gripper` | keyboard ✅ | 4 + 2 fingers | keyboard ✅ | joint / 6D-pose+IK | 0.1 |
| `fahexa750_gripper` | keyboard ✅ | 4 + 2 fingers | keyboard ✅ | joint / 6D-pose+IK | 0.1 |

Verified for all six rows: spawn, hover (`tilt <= 1.1 deg`, holding
within 3 cm of the setpoint), the scripted keyboard sequence
hover→ascend→forward→yaw→release (18-27 deg of forward lean, which is the
under-actuated attitude response and the point of flying them this way), and
every arm DOF reaching a commanded drive target. One caveat on that last
column: `iris650_soarm100`'s `shoulder_pan` is deliberately locked, because
its drive dies whenever any other dynamic body moves — see
`docs/ARM_DISTURBANCE.md`. Every other DOF on every robot reaches its
commanded travel (`hummingbird340_2dof_gripper`'s `R1`/`R2` used to reach
only 4 %; that was the `DriveAPI` instance-name bug, now fixed).

Run any of them with:

```bash
python examples/aerial_manipulator_control_demo.py --preset iris650_soarm100 --mode dynamic --uav_control keyboard --arm_control keyboard
```

Keys: `WASD` translate, `E`/`Q` up/down, `Left`/`Right` yaw; `1`-`9` select an
arm joint, `[`/`]` move it, `G`/`H` close/open every gripper DOF, `Z` home the
arm.

## The scripted mission: `examples/show_am_action.py`

A stand-in for the devices that aren't wired up yet, driving the same setpoint
integration and the same arm write path the keyboard uses:

```
idle -> climb to 1 m -> forward 1 m -> left 1 m -> right 1 m -> yaw 180 deg
     -> forward 1 m -> hold while every drivable joint sweeps +45 deg, back,
        -45 deg, back -> land
```

```bash
python examples/show_am_action.py --preset uaquad600_gripper
python examples/show_am_action.py --preset all        # all six, one scene, together
```

`--preset all` spawns every preset 4 m apart on the same ground plane and flies
them through the mission simultaneously, each with its own controller,
setpoint, residual and arm write mode. Measured numbers come out identical to
the single-robot runs, so they are not interfering with each other.

Translations are body-relative, so the second "forward" retraces the first.
45 deg is clamped per joint **per side** to that joint's own range -- a
two-finger gripper's prismatic DOF gets its few centimetres of travel rather
than being pinned to zero by a symmetric clamp -- and the clamped amplitude is
printed per joint. Every run of the whole fleet:

| preset | arm-sweep worst tilt | mission worst tilt | landed at | write |
|---|---|---|---|---|
| `amov950_airbot_bottom` | 0.50° | 9.26° | 0.478 m | drive |
| `amov950_airbot_dexterous_bottom` | 0.46° | 9.58° | 0.478 m | **teleport** |
| `hummingbird340_2dof_gripper` | 0.74° | 9.70° | 0.379 m | drive |
| `iris650_soarm100` | 0.59° | 8.10° | 0.313 m | drive |
| `uaquad600_gripper` | 0.50° | 8.67° | 0.054 m | drive |
| `fahexa750_gripper` | 0.53° | **0.53°** | 0.134 m | drive |

It also prints the peak travel each joint actually reached during the sweep,
which is the only way to tell a joint that moved and came back from one that
never moved at all -- how hummingbird's undriven arm went unnoticed through
several passes of flight testing.

The "mission worst tilt" is always the ground-contact bounce in the final
phase, not flight; in-flight the worst is the 8-9 deg lean of the
right-after-left reversal. The arm sweep costs **under 0.7 deg on every
robot** -- that column is the isolation working.

`amov950_airbot_dexterous_bottom` needs `--arm-write teleport`: under PD
drive, a *commanded* finger sweep sends the Inspire hand's drives non-finite
within the first sweep phase ("Invalid PhysX transform" on
`right_thumb_3`/`right_index_rubber_2`), even at its 0.001 residual, which is
only enough to hold the hand still. Relevant to the planned Apple Vision Pro
path: that retargeter has to teleport, not drive. Under teleport the fingers
also do not fully return to zero (thumb_3 ends at +18.8 deg, index_2 at
+21.9 deg) -- a teleported joint has no drive restoring it against its
neighbours.

## Several robots in one scene: the view-invalidation trap

Worth knowing before writing any other multi-robot script here, because it
does not raise -- it **segfaults**, inside `get_body_masses`, with no Python
traceback pointing at the cause.

`AerialManipulatorBase.initialize` calls `SimulationContext.instance().reset()`
a second time on purpose: PhysX's first rest-pose solve for an articulation
that `mount_child` has just welded a branch onto is wrong by exactly that
branch's own joint offset, and a second reset re-solves it correctly. That was
already documented in the code. What was not: **the teardown is global.** That
reset destroys the context's physics simulation view, so every view handle any
*other* robot is holding goes stale, and reading one crashes the process.

With one robot it is invisible, which is why it survived this long. With two it
is fatal, and reproducible in about thirty lines: initialize robot A, read its
masses (fine), initialize robot B, read A's masses again, segfault.

So `initialize` now takes `reset_rest_pose=False`, and a multi-robot caller
owns the sequence:

```python
for rig in rigs: rig.am.initialize(reset_rest_pose=False)
sim.reset()                      # once, for the whole scene
for rig in rigs: rig.am.reattach()
```

Two supporting changes in OmniDrones, both one-liners that only matter here:
`RobotBase.initialize` and `MultirotorBase`/`AmovDrone`'s extra body views now
pass `SimulationContext.instance().physics_sim_view` instead of letting each
view call `create_simulation_view` + `set_subspace_roots("/")` for itself --
which was a second, independent way for one robot to invalidate another.

## Four asset defects that only showed up by eye

Every one of these passed the flight metrics -- hover, tilt, position error --
and was caught by somebody looking at the viewport. All four are defects in
vendored assets; all four are now corrected at spawn time, from per-preset
declarations in `robots/presets.py`.

**`hummingbird340_2dof_gripper`: the arm would not move.** `R1`/`R2` ship
`PhysicsDriveAPI:X` -- a `UsdPhysics.DriveAPI` applied under the joint's AXIS
name rather than its KIND. PhysX honours only `angular` on a revolute joint,
so the drive was invisible to the solver: the joints had no effective drive at
all, and both `set_gains` and `set_joint_position_targets` were silent no-ops.
`omni.importer.mjcf` authored it that way. `utils.normalize_joint_drives` now
re-authors it under `angular`, copying the asset's own parameters across
(kp=50, kd=5.0 -- exactly what `manipulators/dynamixel_2r.py`'s docstring had
always claimed were set). `R1`/`R2` now reach the full ±45.5° of a commanded
sweep, against ±0° before. It runs on every preset and is a no-op where the
drive is already correct.

**`uaquad600_gripper`: the arm was mounted underneath.** `quad_scorpion.usd`
hangs its arm below the airframe (measured: every arm body 0.05-0.29 m below
`base_link`), while this project's own MJCF bundle for the same robot --
`ambench_quad_600_top_arm_final`, "arm explicitly flipped 180 deg and
re-mounted top-side-up on request" -- puts it on top, and the preset's comment
has claimed "Mount: top" all along. The preset now re-authors the
`arm_base_fixed` joint's local pose to that bundle's own transform, `pos
(.1, 0, .003)`, `quat (0, 1, 0, 0)`. Arm bodies now sit 0.04-0.24 m *above*
`base_link`, and with nothing hanging below it lands at 0.054 m instead of
0.313 m.

**`fahexa750_gripper`: the arm passed through the airframe.**
`hexa_scorpion.usd` authors `enabledSelfCollisions = False`; the other five
presets all author `True`, and AM-Bench's own `ArticulationCfg` for this
robot asks for `True` (`ambench/robots/fa_hexa.py`). Now set per preset.
Side effect, and the correct one: `arm_link1`/`arm_link2`'s negative travel is
now partly blocked by the body it used to pass through (-15.5° and -5.2°
reached instead of the full -22.9°/-45°).

**`amov950_airbot_dexterous_bottom`: the arm flailed.** Not ground contact --
it has 0.11 m of clearance at the default spawn height. The Inspire hand's
twelve finger drives are authored at kp=0.05..0.07, kd=1e-4, in the same
articulation as arm joints at kp=1e7, kd=1000: a 1e8 spread. Commanding the
fingers through those drives diverges at every residual tried. Scaling the
gains down with the residual -- which is the textbook fix, since a PD drive's
frequency is `sqrt(kp/I)` and the freeze divides `I` by up to 1000 -- was
tried and is *much* worse: the sweep threw the drone 19 m up at 154° of tilt,
against merely going non-finite before, because the scaled-down fingers have
no drive left and self-collide freely. So the gains are left alone and the
preset declares `arm_write="teleport"`, which runs the whole mission at 0.46°
of body tilt. **The planned Apple Vision Pro retargeter has to use that path
too.** Under teleport the fingers do not return cleanly (index_2 peaks at
+88° against a commanded +45°, a teleported joint having no drive to resolve
it against its neighbours' constraints).

## Is the arm isolated on the AM-Bench presets?

Yes, and the mission's arm-sweep phases measure it directly. Both at their
shipped `arm_dynamics_residual = 0.1` against fully coupled (`--arm-residual
1.0`), worst body tilt over the four sweep phases and worst position error:

| | 0.1 (shipped) | 1.0 (coupled) |
|---|---|---|
| `uaquad600_gripper` | **0.65°**, 0.070 m | 3.79°, 0.734 m |
| `fahexa750_gripper` | **0.59°**, 0.017 m | 2.83°, 0.149 m |

About 5x on tilt for both. But for `uaquad600_gripper` the residual buys
something bigger than the dynamic reaction: at 1.0 it carries a **persistent
0.6-0.9 m position error through the entire mission**, arm moving or not
(`idle 0.615 m`, `forward 0.694 m`, `left 0.910 m`). That is the static
moment of its 0.72 kg top-mounted arm, which `LeePositionController` has no
integral term to trim -- the same failure mode as hummingbird's 0.49 m sag,
and the reason AM-Bench's own PID has `ki_pos`. `fahexa750_gripper` does not
show it (0.09-0.15 m throughout), its arm being better centred.

Neither residual has been *swept* for these two presets; 0.1 is verified
adequate, not shown optimal.

## The three ways a joint command can reach the sim

`--arm-write` picks one; `auto` (the default) means teleport for the SO-ARM
leader and drive for everything else.

| | `drive` (PD targets) | `teleport` (write joint state) |
|---|---|---|
| lag | one control step | none, exactly 1:1 |
| disturbance on the drone | ~600x smaller (measured on so_arm100) | kicks the body every step |
| visual flicker | none | yes, stale solver velocity |
| works on so_arm100's `shoulder_pan` | **no** (dead drive) | yes |
| `R1`/`R2` on hummingbird | weak (4 % of target) | yes |

The third way is the one not to use: writing joint *velocities* alongside
either. It flips the drone on the teleport path (148 deg of tilt) and makes
the dexterous hand's drives diverge harder on the drive path (1e12 rad vs
1e4). Both measured; neither is a knob.

## Where the numbers and the locks live

Three layers, each overriding the one before:

1. **the preset** — `AerialManipulatorCfg(arm_dynamics_residual=...,
   arm_locked_joints=...)` in `robots/presets.py`. These are robot
   properties, measured per robot, and they are what `--preset` uses.
2. **the scene** — `task.arm_dynamics_residual` / `task.arm_locked_joints` in
   `cfg/task/<Scene>.yaml`, for `--task`.
3. **the CLI** — `--arm-residual`, `--arm-lock-joints`.

The residual column above is not cosmetic. `amov950_airbot_dexterous_bottom`
**requires** 0.001: at 0.1 and at 1.0 its twelve Inspire-hand finger drives
diverge to 1e4 then 1e12 rad, PhysX reports "Invalid PhysX transform", and the
articulation goes NaN within 50 steps. Full derivations for every value:
`docs/ARM_DISTURBANCE.md`.

## The AM-Bench presets' KF/KM: closed, and the chosen number is a gauge

Both `ambench_600.yaml` and `ambench_750.yaml` now exist, so nothing is
missing -- but it is worth being precise about which numbers were read and
which were derived, because the old note in `robots/drones/ambench_600.py`
rightly refused to fabricate them.

**Read from AM-Bench / the asset**: rotor positions, per-rotor thrust axes,
spin directions, `thrust_limits = (0, 23)` N per rotor,
`reaction_torque_ratio = 0.02` (`ambench/robots/ua_quad.py`, `fa_hexa.py`), and
every body mass and inertia.

**Chosen**: `max_rotation_velocities = 838` rad/s, with `force_constants`
back-solved as `23 / 838^2`. AM-Bench's rotor model commands thrust directly
and has no physical rotor speed to read.

That choice turns out to have **no physical effect at all**. Every consumer
touches the rotor speed only through two products:

| consumer | what it uses | value |
|---|---|---|
| `RotorGroup.__init__` | `KF = omega^2 * force_constant` | 23 N (the real ceiling) |
| `RotorGroup.__init__` | `KM = omega^2 * moment_constant` | ratio x 23 N |
| `LeePositionController` | `max_thrusts = omega^2 * force_constant` | 23 N |
| `compute_parameters` (mixer) | `moment_constant / force_constant` | the ratio alone |

The rotor transient is unaffected too -- `RotorGroup` hard-codes
`tau_up = tau_down = 0.43` in normalized throttle units, with no reference to
rotor speed. `omega` appears *alone* in exactly one place, `multirotor.py`'s
visual rotor spinning, and `RotorlessMultirotor` disables that path
(`rotor_joint_indices = None`) since these assets have no rotor joints to spin.

Verified rather than argued: re-deriving `ambench_600.yaml` at
`omega = 500` rad/s with `force_constants`/`moment_constants` rescaled to keep
the same 23 N and 0.02 gives flight numbers **identical to every printed
digit** (hover 1.96 deg / 0.112 m, forward 18.70 deg / 1.396 m, yaw 16.49 deg,
and the same six arm-drive percentages). So the only genuinely invented number
cancels out algebraically, and the two physically meaningful inputs both come
from AM-Bench.

What is still *chosen rather than tuned* for these two presets is the
controller gains (`lee_controller_ambench_600.yaml` /
`_750.yaml`): scaled to reproduce the stock hummingbird tune's effective
attitude bandwidth against each airframe's inertia. They fly; they are not
optimised. `fahexa750`'s 1.84 thrust/weight in particular leaves less headroom
for the shared position gains than the quad's 2.13.

## What AM-Bench does differently

AM-Bench (`/home/benhua/ambench`, `scripts/environments/teleop_se3_agent.py`)
is IsaacLab-based and makes three different choices:

| | AM-Bench | here |
|---|---|---|
| device layer | `isaaclab.devices.Se3Keyboard`, callback-registered | `KeyboardUAVControl`/`KeyboardArmControl`, carb polling, read once per frame |
| what the keys command | a 7-vector `[dpos(3), drotvec(3), gripper]` integrated into an absolute **end-effector** pose; the drone moves as a consequence | the **drone body**: `(vx, vy, vz, yaw_rate)` integrated into a position+yaw setpoint, with the arm on its own channel |
| controller | PID4DOF / PID6DOF / L1-adaptive / whole-body MPC, **with integral terms** | `LeePositionController`: PD plus an `mg` feedforward, **no integral** |
| mass & inertia | `get_aggregate_physical_parameters` recomputes whole-articulation mass and parallel-axis inertia at runtime | per-drone yaml constants (mass is now corrected to the real articulation total; **inertia is still the bare airframe's**) |
| the arm's reaction | left real; absorbed by the integrators and the whole-body controller | scaled down by `arm_dynamics_residual` |
| actuation | 4-DOF fixed-rotor allocation *and* a 6x12 variable-tilt allocation | 4-DOF and **6-DOF** fixed-rotor allocation; no variable-tilt (no such airframe here) |
| arm commands | the teleop script refuses anything but `ActionMode.ABSOLUTE_EE_POSE`, so EE-centric only -- though the pipeline itself also defines `ABSOLUTE_BASE_JOINTS` (`8 + arm_joint_count`), which is per-joint and simply unused by that script | per-joint, directly |

Two things worth taking from it:

- **Integral terms.** Our sagging and steady-tilt failures are exactly what an
  integrator removes. hummingbird sank 0.49 m purely because its controller's
  mass was 16.5 % low, and `iris650_soarm100` holds a steady tilt whenever its
  arm parks off-centre. An integral term would absorb both *and* much of what
  `arm_dynamics_residual` is currently papering over -- which is the cheaper
  path to "the arm may move freely" than per-robot residual tuning.
- **Runtime aggregate inertia.** Same function, same reason. We now take the
  real total *mass* at `initialize()`; the inertia is still the yaml's
  bare-airframe value, and the arm (plus whatever the residual lumps onto
  `base_link`) changes it.

And the gap in the other direction: AM-Bench's keyboard cannot command a
single joint, which is what you need when checking a new robot's joint limits,
signs, and drive health. That is why the keyboard device here is a joint
selector rather than an EE delta.

## fahexa750_gripper: a fully-actuated airframe, flown both ways

`hexa_scorpion.usd` is a **fixed-cant fully-actuated hexarotor** -- not a
variable-tilt one. Its six rotors sit at fixed, non-coplanar angles
(`FA_HEXA_PROP_RPYS` in `ambench/robots/visuals/propeller_specs.py`), which is
what buys it full 6-DOF wrench authority with no moving parts.

Measured from that geometry:

- thrust axes are canted ~30 deg: `(-0.249, +0.433, 0.866)`,
  `(+0.500, 0, 0.866)`, `(-0.249, -0.433, 0.866)`, repeating for rotors 3-5;
- equal thrust on all six gives net force `(0.003, 0, 5.196)` and net torque
  `(0, -0.0002, 0)` -- i.e. **it hovers on equal thrusts**, the cant cancels;
- the 6x6 wrench matrix has rank 6 and condition number 4.2 -- genuinely,
  cleanly fully actuated;
- only 19.917 N of each rotor's 23 N reaches the vertical, so thrust/weight is
  1.84 against the quad's 2.13.

It flies, and the mixer is **exact**, which is not obvious: `compute_parameters`
builds its 4-row map as `[sin(angle)*arm, -cos(angle)*arm,
-direction*moment/force, 1]`, which looks like it hard-codes "every rotor
thrusts along +Z". It does not constrain the geometry nearly that much. Take
the thrust variable to be each rotor's *vertical force component* and that
family reproduces the true 4-row map of a canted platform exactly:

| yaml field | solved as | why it is free to be solved |
|---|---|---|
| `arm_lengths[i]` | \|tau_xy,i\| | two free parameters (arm, angle) fit an arbitrary 2-vector |
| `rotor_angles[i]` | atan2(tau_x,i, -tau_y,i) | same |
| `moment_constants[i]/force_constants[i]` | \|tau_z,i\| | per-rotor arrays, so each rotor's yaw share fits |
| `directions[i]` | -sign(tau_z,i) | works out to -spin_dir here |
| `force_constants[i]` | 23 * axis_z,i / 838^2 | the thrust variable is the vertical component |

where `tau_i = r_i x axis_i + drag_i` per unit vertical thrust. Reconstruction
residual against the true per-rotor torques: **5.6e-17**. Two conventions carry
it (both in `robots/drones/rotorless_multirotor.py`): `rotor_axes_b` is scaled
to z-component 1 rather than unit length, and the *physical* drag ratio moves
to `rotor_drag_ratio` because the yaml's `moment_constants` now also carry the
lever arm's own yaw contribution -- which a canted rotor has and a +Z rotor
does not.

What that mixer gives up is **capability, not correctness**: the lateral force
those canted rotors could produce at a level attitude. A 4-DOF controller never
asks for it, so under `--controller lee` this platform tilts to translate like
any underactuated multirotor (measured 17.9 deg of forward lean, right
alongside the quad's 18.7).

### The 6-DOF allocator

`controllers/drone/FullyActuatedPositionController` (default for this preset,
`--controller auto`) uses that authority. Same shape as AM-Bench's
`pid_6dof_ctrl` + `controllers/utils/control_allocation.py`: force and attitude
channels are independent (`force_b = R^T force_w`, with the attitude target
commanded separately) rather than the geometric `b_3d = A/||A||` that *defines*
the attitude from the desired acceleration. Column `i` of the 6xN matrix is
`[axis_i ; cross(r_i, axis_i) - dir_i * drag_ratio * axis_i]`, read off this
plant's own `apply_action` rather than re-derived, so allocation and simulation
cannot disagree about a sign. Differences from AM-Bench's version: no integral
terms (the yaml's PD gains are reused as-is), and no 6x12 variable-tilt branch
(no actuated-tilt airframe here).

The lateral command is clamped to **15.69 N (0.241 g)** -- the worst-azimuth
horizontal force holdable at hover with every rotor still inside
`[0, max_thrust]`. Past that the allocation stops being exact and PhysX
silently clips the wrench, which couples back into attitude.
`allocation.py`'s own self-check re-derives that number and asserts it; it
runs without Isaac Sim.

Same mission, same gains, only `--controller` changed:

| phase | `lee` | `fa` |
|---|---|---|
| forward 1 m | 4.12° | **0.20°** |
| left 1 m | 4.28° | **0.22°** |
| right 1 m | 8.09° | **0.20°** |
| yaw 180° | 2.95° | **0.23°** |
| arm sweep | 0.59° | 0.53° |
| **worst over the mission** | **8.09°** | **0.53°** |
| position error while translating | 0.303 m | 0.320 m |
| settled position error | 0.058 / 0.073 m | **0.034 / 0.038 m** |

The remaining 0.53 deg is the arm sweep, not flight. Translating costs a
slightly larger tracking error, which is the lateral clamp doing its job; the
settled error is tighter.

One unresolved oddity, inherited: the asset's authored `base_link` inertia
`(0.198, 0.198, 0.460)` violates the triangle inequality (`Izz > Ixx + Iyy`).
The MJCF export had to clamp `Izz` to 0.395604 to compile in MuJoCo. PhysX
accepts it, and `ambench_750.yaml` keeps the authored value so the controller's
gains match the body actually being simulated -- but the number is not
physical.

`uaquad600_gripper`, by contrast, is a plain +Z X-quad, where the yaml's
parameters are the airframe's own and `moment_constants/force_constants` really
is the 0.02 drag ratio (`r x z_hat` has no z-component). Both presets need
`robots/drones/rotorless_multirotor.py`, which supplies rotor geometry
statically because AM-Bench's USDs have no rotor rigid bodies at all -- their
propellers are visuals under `base_link/visuals/`.
