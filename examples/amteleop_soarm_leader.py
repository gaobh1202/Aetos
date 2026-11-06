"""Teleop framework: drive an aerial manipulator's SO-ARM-family arm with a
real SOArmLeader (lerobot SO101/SO-ARM100 leader, read over USB-serial).

This is the generalized, extensible successor to the one-off
scripts/teleop_so101_leader.py built while debugging this pipeline. For now
it only wires up what that script already proved out -- default scene,
single robot (``iris650_soarm100``), 1:1 arm teleop, drone body
pinned in place -- but the pieces are split out so the planned follow-ups
slot in without a rewrite:

  - more scenes: extend ``build_scene()``
  - drone flight control from the same leader/another device: extend
    ``Teleop.step()`` where ``drone_action`` is currently a constant zero
  - data recording: extend ``Teleop.step()`` where the TODO marks it, using
    ``joint_state``/``radians``/the robot's own joint positions each step

Usage (real hardware):
    python examples/amteleop_soarm_leader.py --port /dev/ttyACM0

Usage (no hardware connected -- exercises the sim side only):
    python examples/amteleop_soarm_leader.py --fake

See docs/TELEOP.md for setup (serial permissions, calibration, the
ISAACSIM_PATH env var this needs in a fresh shell).
"""
import argparse
import math
import time

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--robot", type=str, default="iris650_soarm100",
    help="Preset name from aetos_multiam.robots.presets.PRESETS. Must have an SO-ARM-family "
    "manipulator (manipulator_type containing 'so_arm') -- this script 1:1-mirrors the leader "
    "onto SO-ARM joint names, so it can't drive anything else.",
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the SOArmLeader.")
parser.add_argument("--recalibrate", action="store_true", help="Force leader recalibration even if cached.")
parser.add_argument(
    "--fake", action="store_true",
    help="Skip the real leader device entirely; drive joints with a slow sine sweep instead "
    "(for testing the sim side without hardware connected).",
)
parser.add_argument("--rate", type=float, default=30.0, help="Target control loop rate (Hz).")
parser.add_argument("--headless", action="store_true", help="Run without a GUI window.")
parser.add_argument(
    "--max_steps", type=int, default=0,
    help="Stop after this many control steps (0 = run until the window closes / Ctrl+C).",
)
args = parser.parse_args()

# Connect/calibrate the leader BEFORE opening the Isaac Sim GUI window:
# first-run calibration is interactive (input() prompts + a live-updating
# console table) and can take a while -- doing it after the window is
# already open leaves the GUI's event loop unserviced that whole time,
# which the window manager reports as "not responding". Nothing below this
# needs Isaac Sim.
leader = None
if not args.fake:
    from aetos_multiam.devices.soarm_leader import SOArmLeader

    leader = SOArmLeader(port=args.port, recalibrate=args.recalibrate)

simulation_app = init_simulation_app({"headless": args.headless})

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as physx_script_utils
import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdPhysics

import omni_drones.utils.kit as kit_utils
from aetos_multiam.devices.action_convert import JOINT_ORDER, convert_leader_state_to_radians
from aetos_multiam.devices.soarm_leader import MOTOR_LIMITS
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase


def require_soarm_manipulator(preset_name: str) -> None:
    """This script's whole control scheme (SOArmLeader -> JOINT_ORDER names,
    direct 1:1 position mirroring) assumes an SO-ARM-family manipulator.
    Fail loudly and early for any preset that doesn't have one, rather than
    spawning something the leader can't actually drive."""
    cfg = get_preset(preset_name)
    manipulator_type = (cfg.manipulator_type or "").lower()
    if "so_arm" not in manipulator_type and "soarm" not in manipulator_type:
        raise ValueError(
            f"Preset '{preset_name}' has manipulator_type={cfg.manipulator_type!r}, "
            "which isn't an SO-ARM-family manipulator. amteleop_soarm_leader.py mirrors the "
            "SOArmLeader 1:1 onto SO-ARM joint names and can't drive anything else -- pick a "
            "preset whose manipulator_type contains 'so_arm' (e.g. 'iris650_soarm100')."
        )


def build_scene(sim: SimulationContext) -> None:
    """Ground plane + lights, gravity zeroed. Extension point for later:
    walls, objects, multiple robots -- add them here, keyed by a future
    --scene argument."""
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))
    prim_utils.create_prim("/World/Light/GreySphere2", "SphereLight", translation=(-2.0, -2.0, 5.0))
    physics_scene_path = sim.get_physics_context().prim_path
    UsdPhysics.Scene(prim_utils.get_prim_at_path(physics_scene_path)).CreateGravityMagnitudeAttr(0.0)


def pin_base_link_to_world(prim_path: str) -> None:
    """Weld base_link (the drone body) to the world with a real PhysX joint,
    done before sim.reset() so it's part of the articulation from the start.

    Re-teleporting the drone's pose every step (the first approach tried)
    doesn't work: the merged drone+arm+gripper articulation's actual PhysX
    root turns out to be a body deep in the arm chain, not base_link, despite
    base_link being body0/parent of every mount joint per usd_compose
    .mount_child's own createJoint() calls (confirmed empirically: body_names
    came back with an arm link at index 0). Teleporting "the root" pose pins
    that arm link instead and leaves base_link free to swing as a consequence
    of the arm's own joint motion. A world-anchored FixedJoint directly on
    base_link sidesteps the root-identity question entirely.
    """
    physx_script_utils.createJoint(
        prim_utils.get_current_stage(), "Fixed", None, prim_utils.get_prim_at_path(f"{prim_path}/base_link")
    )


def fake_joint_state(t: float) -> dict:
    """Sine-sweep each joint across ~60% of its leader-normalized range, for
    --fake mode. Gripper sweeps 0..100 (its own, non-symmetric, range)."""
    arm_swing = 60.0 * math.sin(t)
    return {
        "shoulder_pan": arm_swing,
        "shoulder_lift": arm_swing,
        "elbow_flex": arm_swing,
        "wrist_flex": arm_swing,
        "wrist_roll": arm_swing,
        "gripper": 50.0 + 50.0 * math.sin(t * 0.7),
    }


class RateLimiter:
    """Sleeps (rendering meanwhile) to hold a target loop rate -- same pattern
    as leisaac's own teleop_se3_agent.py RateLimiter."""

    def __init__(self, hz):
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)
        self.last_time = time.time()

    def sleep(self, sim):
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            sim.render()
        self.last_time = max(self.last_time + self.sleep_duration, time.time() - self.sleep_duration)


class Teleop:
    """One robot, teleoperated by one leader device. Owns the per-step
    control logic; scripted at module level around it for now since there's
    only one robot/one scene -- promote to real multi-robot plumbing if/when
    a scene actually needs more than one."""

    def __init__(self, sim: SimulationContext, preset_name: str, prim_path: str, translation=(0.0, 0.0, 1.0)):
        require_soarm_manipulator(preset_name)
        self.sim = sim
        self.prim_path = prim_path
        self.am = AerialManipulatorBase(get_preset(preset_name), name=preset_name)
        self.am.spawn(translations=[translation], prim_paths=[prim_path])
        pin_base_link_to_world(prim_path)

    def initialize(self):
        self.sim.reset()
        self.am.initialize(prim_paths_expr=self.prim_path)
        # Cheap insurance: AerialManipulatorBase.initialize()'s own second
        # SimulationContext reset (needed for the merged articulation's
        # rest-pose solve) is now re-initializing the view itself at the
        # source (see robot_base.py) -- this extra step just guards against
        # the same class of issue if that ever changes again.
        self.sim.step(render=False)
        self.arm_and_gripper_indices = torch.cat([self.am.arm_joint_indices, self.am.gripper_joint_indices])
        self.drone_action = torch.zeros(1, self.am.drone.num_rotors, device=self.am.drone.device)

    def step(self, joint_state: dict, motor_limits: dict):
        radians = convert_leader_state_to_radians(joint_state, motor_limits)
        arm_targets = torch.tensor(
            [[radians[n] for n in JOINT_ORDER]], device=self.am.drone.device, dtype=torch.float32
        )

        # TODO(drone control): drive the drone from here once a second
        # input device (or a second leader channel) exists; self.drone_action
        # is a constant zero-thrust command for now, matched by the world
        # FixedJoint above that holds base_link in place regardless.
        self.am.apply_action(self.drone_action)

        # Direct position teleport, not PD position targets: the goal is a
        # true 1:1 leader->sim mapping, and PD tracking fought that two ways
        # -- the SO101 asset's shoulder_pan joint has an implicit drive that
        # outputs zero torque regardless of gains (confirmed via isolated
        # testing: position target registers correctly, but joint position
        # and velocity both stay completely flat -- a real defect in that
        # one joint's drive, not a tuning issue), and even the joints whose
        # drive does work showed visible settle-time sway on every target
        # change. Teleporting is lag-free and sidesteps both.
        self.am.drone._view.set_joint_positions(
            arm_targets.reshape(1, 1, -1), joint_indices=self.arm_and_gripper_indices
        )
        # Deliberately NOT writing a joint velocity alongside this, even
        # though it would remove the arm's visible per-frame flicker (stale
        # solver velocity gets integrated past the pose just written, then
        # snapped back). Zeroing the velocity dumps the arm's angular momentum
        # into the drone body, and even a commanded finite difference costs
        # real attitude quality -- see the measured table in
        # aerial_manipulator_control_demo.py's own arm-write block. Harmless
        # here while the base is pinned, but keep the two teleop paths
        # identical so a fix lands in both.

        # TODO(data recording): append (joint_state, radians, the robot's
        # own joint positions/velocities, wall-clock timestamp) to a buffer
        # here and flush it on exit, once there's a format to write.

        return arm_targets


try:
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1 / 100,
        rendering_dt=1 / 100,
        backend="torch",
        physics_prim_path="/World/PhysicsScene",
        device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
    )
    build_scene(sim)

    PRIM_PATH = f"/World/{args.robot}"
    teleop = Teleop(sim, args.robot, PRIM_PATH)
    teleop.initialize()

    vp = get_active_viewport()
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
    cam_state.set_position_world(Gf.Vec3d(1.6, -1.6, 2.9), True)
    cam_state.set_target_world(Gf.Vec3d(0.0, 0.0, 2.3), True)
except Exception:
    # Kit's sys.excepthook otherwise swallows this silently -- see the same
    # note on the main control loop below.
    import traceback

    traceback.print_exc()
    simulation_app.close()
    raise

if args.fake:
    print("[amteleop] running with --fake: sine-sweeping the arm instead of reading a real SOArmLeader.", flush=True)

rate_limiter = RateLimiter(args.rate)
t0 = time.time()

step = 0
try:
    while simulation_app.is_running():
        if leader is not None:
            joint_state = leader.read_joint_state()
            motor_limits = leader.motor_limits
        else:
            joint_state = fake_joint_state(time.time() - t0)
            motor_limits = MOTOR_LIMITS

        arm_targets = teleop.step(joint_state, motor_limits)
        sim.step(render=True)
        step += 1
        if step % 30 == 0:
            print(f"[amteleop] step {step}, arm_targets(rad)={arm_targets.tolist()}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
        rate_limiter.sleep(sim)
except KeyboardInterrupt:
    pass
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    if leader is not None:
        leader.disconnect()
    simulation_app.close()
