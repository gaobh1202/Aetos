# Arm-on-drone disturbance: the two knobs, and why the two robots need different values

A mounted manipulator that moves while the drone hovers kicks the drone.
Both aerial-manipulator scenes therefore declare two knobs, read by
`examples/aerial_manipulator_control_demo.py` and overridable on the
command line:

| key (in `cfg/task/<Scene>.yaml`) | ShelfBucketGrasp<br>(`iris650_soarm100`) | ShelfButtonGrasp<br>(`amov950_airbot_bottom`) | AerialHandover<br>(`hummingbird340_2dof_gripper`) | CLI override |
|---|---|---|---|---|
| `arm_dynamics_residual` | `0.1` | `0.001` | `0.1` | `--arm-residual` |
| `arm_locked_joints` | `[shoulder_pan]` | `[]` | `[]` | `--arm-lock-joints` |

(`AerialHandover` is a two-agent collaborative scene and the demo script that
reads these keys drives one robot, so there they are declarative for now --
the measurements behind them were taken on the preset directly.)

`arm_dynamics_residual` is the fraction of each arm link's mass and inertia
left on the link; `1 - residual` is lumped onto `base_link` with a
parallel-axis correction (`aetos_multiam/utils/arm_dynamics.py`). The arm
keeps its full kinematics and collision geometry, and the total mass and
centre of mass of the whole robot are preserved -- only the arm's ability
to react on the body is scaled down. Set `residual: 1.0` (or
`--arm_dynamics coupled`) to restore the true coupled dynamics; that is the
intended end state once a learning-based controller handles the coupling,
and the whole point of making it one number per scene.

`arm_locked_joints` holds a joint at rest instead of following the operator.

## Measured: ShelfButtonGrasp / airbot at the shipped `residual: 0.001`

Per-joint, each joint swept at 40 % of its own half-range (capped 0.6 rad)
at 0.25 Hz, teleport writes, Lee controller holding position:

| joint | yaw | pitch | position |
|---|---|---|---|
| `joint1` (base yaw) | 3.17° | 3.70° | 0.120 m |
| `joint2` | 0.79° | 1.96° | 0.138 m |
| `joint3`..`joint6`, `endleft`, `endright` | ≤0.09° | ≤0.15° | 0.0921 m |
| all eight together | 1.36° | 0.12° | 0.0921 m |

**0.0921 m is an arm-independent baseline** (the hover's own post-reset
settling drift) -- every quiet joint lands on exactly that number. So
`joint1`/`joint2` really contribute only ~0.03/0.05 m.

Un-frozen (`residual: 1.0`), the same `joint1` sweep gives **196° of yaw and
4.10 m** -- the drone flips. The knob is worth ~60x on that joint.

All-joints sweep vs residual, same setup: `1.0 -> 269°`, `0.1 -> 98°`,
`0.05 -> 44.7°`, `0.02 -> 18.5°`, `0.005 -> 18.5°`, `0.001 -> 4.5°`,
`0.0005 -> 2.9°`, `0.0001 -> 0.71°`. Monotonic, no cliff.

## Why iris650_soarm100 needs 0.1 and a locked joint instead

The two robots' disturbances have different *causes*, so the same knob
behaves differently. Arm-to-drone mass ratio is **not** the explanation --
so_arm100 is 0.669 kg of arm on a 3.418 kg body (19.6 %), airbot is 3.00 kg
on 9.85 kg (30.5 %); comparable.

**so_arm100's disturbance is mostly a teleport artifact, not physics.**
Same `shoulder_pan` sweep (0.6 rad @ 0.25 Hz, residual 0.1):

| arm write mode | yaw | tilt | position |
|---|---|---|---|
| `set_joint_positions` (teleport, what we do today) | 67.84° | 27.88° | 0.643 m |
| `set_joint_position_targets` (PD, kp=200) | 0.11° | 0.12° | 0.029 m |
| teleport to a *constant* value every step | 2.4° | -- | -- |

Changing only the write mode buys 600x, and teleporting a *stationary*
target costs 2.4° against 67.8° for a moving one: the cost comes from the
positional state write itself (PhysX turns a position jump carrying a stale
joint velocity into an impulse on the root body), not from the arm's
inertia. A 0.669 kg arm whose heaviest link is 0.162 kg cannot physically
yaw a 3.418 kg drone by 68°.

**airbot's disturbance is real physics.** PD drive and teleport measure the
same there (267/130/15° vs teleport 269/98/18.5° at residual 1.0/0.1/0.02),
so there is no artifact component to remove. Its arm is 3.00 kg with a
0.646 kg heaviest link -- 4x so_arm100's.

Everything else follows:

- Real reaction torque scales with the arm's mass and inertia, so the knob
  is linear on airbot -> **monotonic, and 0.001 gets under 3°.**
- A teleport impulse is velocity-driven, not mass-driven, so lowering the
  residual on iris does not shrink it, while it *does* blow up the mass
  ratio between `base_link` and the now near-massless arm links and leaves
  the articulation ill-conditioned for the TGS solver -> **non-monotonic:
  0.3 -> 2.4° of tilt, 0.1 -> 1.3, 0.02 -> 9.9, 0.005 -> 84.** 0.1 is the
  floor. Do not lower it without re-measuring.
- iris cannot simply switch to PD drive, which is the real fix: whenever any
  other dynamic rigid body in the scene is moving, `shoulder_pan` alone
  stops responding to its joint drive entirely (Isaac Sim 4.1.0 / PhysX
  defect, characterised but not explained -- see the note below). It can
  only be teleported, and teleporting it is what kicks the drone. Hence the
  lock.
- `shoulder_pan`'s axis is pure yaw (`-0.05, +0.01, -1.00`), and yaw is a
  quadrotor's weakest axis (rotor drag torque only), so that artifact
  impulse lands exactly where the controller has least authority. airbot's
  `joint1` is also a base-yaw joint, but its impulse is real and small.

With the lock in place, a full fake-leader sweep moves `cmd_shoulder_pan`
not at all while the other joints swing ±1.047 rad: **drone yaw 0.84°
peak-to-peak (was 56.4°), tilt ≤0.72°.**

## hummingbird340_2dof_gripper: two prerequisite bugs, then an easy knob

This preset could not hover at all before the arm was even moved, and both
causes were in shared code, not in the asset:

**1. The articulation root landed on an arm link.** Most drone USDs here
author `ArticulationRootAPI` on `base_link`; hummingbird's authors it on the
USD's default prim, which Isaac merges onto the spawn root. Harmless for the
bare drone -- PhysX picks `base_link` anyway, being the body the rotor joints
hang off -- but once `mount_child` welds an arm branch under the same
ancestor, PhysX is free to pick an arm link as the floating base, and it
picked `first_segment`. With the arm held *perfectly still* the drone tumbled:
**187.6 deg of yaw, 21.1 m of drift.** Fixed in
`AerialManipulatorBase._pin_articulation_root_to_drone`, which moves the API
onto the drone's mount body (and re-applies the solver settings
`RobotBase.spawn` had authored on the old prim). No-op for `iris650_soarm100`
and `amov950_airbot_bottom`, which already carry it on `base_link`.

**2. The controller's mass did not include the arm.** `params["mass"]` comes
from the bare drone's yaml -- 0.716 kg -- while the mounted arm + gripper make
the real vehicle 0.858 kg, a 16.5 % deficit that `LeePositionController` puts
straight into its gravity feedforward. Measured as a **0.491 m vertical sag**,
identical in every case, arm moving or not. Fixed in
`AerialManipulatorBase.initialize`, which sets `params["mass"]` to the
articulation's real total.

| | yaw | roll | pitch | position |
|---|---|---|---|---|
| arm held still, before either fix | 187.60° | 147.34° | 79.02° | 21.13 m |
| after the articulation-root fix | 0.00° | 0.20° | 0.00° | 0.56 m (-0.491 z) |
| after the mass fix too | 0.00° | 0.30° | 0.01° | **0.003 m** |

With a clean hover underneath, the arm's own disturbance is small and the knob
is linear (all five joints sweeping, teleport writes):

| residual | roll | position |
|---|---|---|
| 1.0 (fully coupled) | 8.27° | 0.510 m |
| 0.3 | 2.19° | 0.153 m |
| **0.1 (shipped)** | **0.72°** | **0.052 m** |
| 0.01 | 0.07° | 0.005 m |
| 0.001 | 0.01° | 0.0005 m |

Per-joint at residual 1.0: `R1` 5.91° roll / 0.363 m, `R2` 2.64° / 0.178 m,
the three fingers ~0. Note the axis: **this arm disturbs roll, not yaw.**
`dynamixel_2r` has no base-rotation joint at all (two pitch hinges `R1`/`R2`
plus three fingers), so so_arm100's `shoulder_pan` failure mode cannot occur
here by construction -- measured yaw is <=0.09° on every joint, and every
joint drives identically with and without a moving dynamic body in the scene.

This is the one robot of the three that barely needs the knob: fully coupled
it already costs only 8.3° of roll, against airbot's 269° and so_arm100's
67.8°. `0.1` is shipped as a comfortable default that still keeps a tenth of
the real coupling.

### Side effect on amov950_airbot_bottom

The mass fix is generic, so it also corrected `amov950_airbot_bottom`, whose
yaml overstated its mass by 3.6 % (13.332 declared vs 12.873 real). Its
ShelfButtonGrasp all-joints sweep moved from `1.36° yaw / 0.12° pitch /
0.0921 m` to `3.48° / 2.01° / 0.0448 m` -- position error halved, attitude
slightly worse, both small. `iris650_soarm100` is unaffected (its own
`initialize` already set the true total, measured 0.0 % error).

### Known, not fixed: PD drive barely moves R1/R2

Commanding `R1`/`R2` to 0.300 rad with kp=200/kd=2.0 for 200 steps reaches
only 0.124 (coupled) or 0.027 (residual 0.1), while the three fingers hit
0.300 exactly. Identical with and without a moving dynamic body, so this is
*not* the so_arm100 drive defect. Not chased down -- the demo teleports the
arm, so nothing currently depends on it -- but anyone wanting PD-driven
teleop on this arm (which is the better write mode, see above) has to solve
this first. Self-collision jamming against the drone body or the gripper is
the first thing to check.

## The `shoulder_pan` drive defect

Minimal repro: drone + so101 mounted, hand-built scene, 200-step PD test,
target 0.300 rad, gains forced to kp=200/kd=2.0 and read back. Adding one
extra dynamic rigid body that *moves* (a falling cube 50 m away, or one with
gravity disabled given `vx = 3 m/s`) kills `shoulder_pan`'s drive; the same
cube held still leaves it working. Affects only that one DOF and only its
drive -- `set_joint_positions` still moves it, and every other arm joint
drives normally. Not GPU-specific, not a naming or joint-indexing problem,
not the fixed-joint mount boundary (all four were tested and ruled out).
airbot's eight joints are unaffected: with and without a moving dynamic
body, every joint drives to the same value. Worth an upstream report.

## Caveats

- Every number above comes from scripted deterministic sweeps
  (`--arm-fake`), never from real leader hardware. Real-leader velocities
  and accelerations differ, so the tuned residuals should be re-measured
  against hardware.
- airbot has no manipulator teleop channel in the demo --
  `--arm_control soarm_leader` rejects non-SO-ARM manipulators -- so
  `arm_locked_joints` is currently inert for ShelfButtonGrasp.
- When probing, a catastrophic case poisons later `env.reset()`s in the same
  process: the `residual: 1.0` run reported 1.5° for the all-joints case,
  smaller than `joint1` alone. Only the first case in a loop runs on a clean
  env. Give each configuration its own process, or read only the first row.
- `droll ≈ 360°` in those logs is an angle-wrap artifact (the reset attitude
  has its roll reference at ±π) and means 0, not a flip.
