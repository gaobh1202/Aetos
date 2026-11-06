# Teleoperating an SO-ARM-family arm with a real SOArmLeader

`examples/amteleop_soarm_leader.py` drives the manipulator joints of an
aerial-manipulator preset 1:1 from a physical lerobot SO101/SO-ARM100
leader arm, read over USB-serial (Feetech STS3215 motors). The drone body
is welded to the world (gravity zeroed) -- this is for teleoperating/testing
the arm, not flight (drone flight control is a planned follow-up, see the
script's own module docstring for the extension points).

By default it uses the `iris650_soarm100` preset; any preset whose
manipulator isn't SO-ARM-family (`manipulator_type` containing `so_arm`)
is rejected with a clear error at startup, since the leader-to-sim mapping
is a direct per-joint mirror onto SO-ARM joint names.

Adapted from [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac),
which implements the same leader -> SO101 follower teleop for IsaacLab. This
project uses raw `omni.isaac.core` (no IsaacLab), so only the hardware-facing
pieces were ported:

- `aetos_multiam/devices/lerobot_motors/` -- the Feetech serial-protocol
  driver, vendored verbatim from leisaac (itself copied from HuggingFace
  [lerobot](https://github.com/huggingface/lerobot)), Apache-2.0. Pure
  `pyserial`/`deepdiff`/`feetech-servo-sdk` -- no sim coupling, so vendoring
  it was lighter than depending on the full `lerobot` pip package.
- `aetos_multiam/devices/soarm_leader.py` -- `SOArmLeader`: connect/
  calibrate/read, same logic as leisaac's own `SO101Leader` minus its
  IsaacLab keyboard-event base class (this script drives its own loop).
- `aetos_multiam/devices/action_convert.py` -- the leader-reading ->
  sim-joint-radians remap (same linear rescale leisaac uses: leader's own
  normalized range -> follower USD's authored degree range -> radians).
  `SO101_USD_JOINT_LIMITS_DEG` was verified directly against this project's
  own `so101_follower.usd` (not just copied from leisaac's docs).

## Setup

```bash
pip install pyserial deepdiff feetech-servo-sdk
```

Grant serial port permissions (either works):

```bash
sudo usermod -aG dialout $USER   # then log out/in for it to take effect
# or, as a one-off:
sudo chmod 666 /dev/ttyACM0
```

Every Isaac Sim script in this project (this one included) needs
`aetos_multiam`/`omni_drones` on the interpreter's path, which means running
via the `omnidrone` conda env with `ISAACSIM_PATH` exported first (a plain
`conda activate omnidrone` in a fresh shell won't pick it up -- its own
activation script silently no-ops without that env var set):

```bash
export ISAACSIM_PATH=/home/benhua/isaacsim
conda activate omnidrone
```

(Isaac Sim's own `python.sh` is *not* an alternative here -- that
interpreter doesn't have `aetos_multiam`/`omni_drones` installed.)

## Running

```bash
python examples/amteleop_soarm_leader.py --port /dev/ttyACM0
```

First run (or with `--recalibrate`) walks through the leader's own
calibration interactively in the terminal: move to the center of its range
and press ENTER, then sweep every joint through its full range and press
ENTER again. The result is cached at
`aetos_multiam/devices/.cache/soarm_leader.json` (gitignored).

To exercise the sim side without the physical leader connected (sine-sweeps
the arm instead):

```bash
python examples/amteleop_soarm_leader.py --fake
```

To use a different SO-ARM-family preset:

```bash
python examples/amteleop_soarm_leader.py --robot <preset_name> --fake
```

## Notes

- Leader and follower (SO101 substitute, see `so_arm100.py`'s module
  docstring) are the same hardware family, so this is a direct per-joint
  mirror -- no IK involved.
- The arm is driven by **direct joint-position teleport**
  (`set_joint_positions`), not PD position targets. Two reasons: the SO101
  asset's `shoulder_pan` joint has an implicit PD drive that outputs zero
  torque regardless of gain tuning (confirmed via isolated testing -- a real
  asset defect, not something fixable from this script), and even the
  joints whose PD drive does work showed visible settle-time sway on every
  target change. Teleporting is lag-free and sidesteps both, which also
  matches the explicit goal here: an exact 1:1 leader<->sim mapping, not
  physically-damped tracking. Trade-off: the arm can't be blocked by
  collisions or exert realistic contact forces this way -- revisit if a
  later scene needs the arm to push against real objects.
- The drone body is welded to the world with a real PhysX `FixedJoint`
  (`pin_base_link_to_world()` in the script), not by re-teleporting its pose
  every step -- the merged articulation's actual PhysX root turned out to be
  a body inside the arm chain, not `base_link`, so pose-teleporting "the
  root" pinned the wrong body. See the function's docstring for the full
  story.
- `--rate` (default 30 Hz) paces the control loop; the leader's own serial
  read latency (~6 motors over one bus) already limits this somewhat in
  practice.
- Kit's own `sys.excepthook` silently swallows uncaught Python exceptions in
  these scripts (clean exit, no traceback) -- the script wraps its risky
  sections in `try`/`except` with `traceback.print_exc()` because of this;
  keep that pattern in any new Isaac Sim script in this project.

## Related: viewing any registered preset

`examples/show_robot.py` opens an interactive window showing one preset (or
all of them side by side) -- useful for a quick "does this preset's USD
exist and mount correctly" check, independent of teleop:

```bash
python examples/show_robot.py --robot iris650_soarm100
python examples/show_robot.py --robot all
python examples/show_robot.py --robot all --headless   # screenshot only
```
