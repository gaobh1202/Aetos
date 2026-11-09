"""Teleoperate an aerial manipulator: drone body and arm, each from its own device.

Two independent control sources, picked separately:

  --uav_control  none (default) | keyboard
  --arm_control  none (default) | soarm_leader

"none" means that half just holds still -- the drone keeps station on its
spawn setpoint, the arm holds its rest pose. So the bare `--task X` run is a
robot hovering in a scene doing nothing, and you opt each half into being
driven. The previous keyboard-only behaviour of this script (it used to be
``keyboard_uav_control_demo.py``) is now ``--uav_control keyboard``.

``--arm_dynamics`` decides whether the arm pushes back. The default,
``frozen``, keeps the arm's static mass and inertia (lumped onto ``base_link``)
but makes the links themselves effectively massless, so swinging the arm
transmits no reaction wrench into the drone body and a controller tuned for a
fixed airframe stays stable -- the same "simplify the plant first, learn the
real coupled dynamics later" split humanoid pipelines use between retargeting
and the learned controller. ``--arm_dynamics coupled`` restores the real
multibody physics; expect a hand-tuned Lee controller to diverge under fast
arm motion there. ``aetos_multiam/utils/arm_dynamics.py`` spells out exactly
what is and isn't preserved.

``--arm_control soarm_leader`` reads a real lerobot SO101/SO-ARM100 leader
over USB-serial and mirrors it 1:1 onto the follower arm, the same mechanism
and the same hardware path ``examples/amteleop_soarm_leader.py`` proved out
(that script stays as the arm-only, drone-pinned version; this one is the
combined drone+arm rig). ``--arm-fake`` sine-sweeps the joints instead, for
testing with no hardware connected. The leader is connected and calibrated
BEFORE Isaac Sim boots -- first-run calibration is interactive and would
otherwise leave the GUI event loop unserviced long enough for the window
manager to call it hung.

``aetos_multiam.devices.keyboard_uav_control.KeyboardUAVControl`` is just the
input device (polls WASD/E/Q/arrows -> a normalized ``{vx, vy, vz,
yaw_rate}`` intent each frame); this script is the driver loop plus BOTH
flight-control implementations wired to it -- same device/driver split
``amteleop_soarm_leader.py`` uses for its leader arm.

By default it spawns a bare OmniDrones drone directly via
``MultirotorBase.make()``/``drone.spawn()`` (the same proven path OmniDrones'
own ``examples/00_play_drones.py`` uses) -- no arm or gripper involved, so the
composition machinery buys nothing.

``--task <TaskName>`` is the one to reach for normally: it builds a whole
registered scene (``cfg/task/<TaskName>.yaml`` -> ``environments.make()``) --
arena, props and all -- and drives **the robot that scene itself declares**, so
the robot choice lives with the scene instead of being retyped on the command
line. ``AerialBucketGrasp``, for instance, declares
``aerial_manipulators: [{preset: amov950_airbot_bottom}]``, and that is what
you end up flying around its shelf. The scene is built and then driven straight
off ``SimulationContext.step()``: this is teleop, so the env's own
step/reward/episode-reset machinery is deliberately bypassed (otherwise the
episode timer would yank the drone back to spawn mid-flight). The env is used
as a scene builder and a robot registry, nothing more.

A scene that welds its robot to the world needs to be told not to -- this
script passes ``task.weld_robot=false``, which ``AerialBucketGrasp`` honours
(it defaults to false there now, but passing it costs nothing and keeps the
script working against a scene that welds).
A scene without that switch will simply not move.

``--preset <name>`` is the bare version of the same thing: one aerial
manipulator (drone + arm [+ gripper]) over an empty ground plane, no scene.
Two things differ on both of those paths and both are load-bearing:

* The arm's joints are held at their rest pose every single step
  (``hold_arm_joints(..., zero_velocity=True)``). An unheld USD joint drive
  with no refreshed position target goes numerically unstable on its own, and
  the reaction torque transmits straight into the drone body through the
  shared articulation -- it tumbles the drone no matter what the flight gains
  are. See ``lee_controller_amov_drone_arm_1.yaml``'s own writeup.
* ``LeePositionController`` is built with the literal ``9.81``, not
  ``drone.gravity``: ``MultirotorBase.initialize()`` overwrites that attribute
  with the robot's WEIGHT in newtons, and this path forces controller
  construction to happen after ``initialize()`` (the ArticulationView has to
  exist first).

Both modes share one integration step: the keyboard intent integrates into
a moving ``target_pos``/``target_yaw`` setpoint (world frame; vx/vy are
rotated by the CURRENT target yaw before being added, so "forward" always
means "forward relative to where the drone is currently facing"). The two
modes differ only in how that setpoint gets realized:

--mode kinematic (default): ignores the drone's own rotor dynamics
entirely -- ``target_pos``/``target_yaw`` is teleported straight onto the
drone's body every step (``set_world_poses`` + zeroed ``set_velocities``, so
real scene gravity has no visible effect). This is "fly a fixed point in
space": movement is instant in whatever direction is held, body attitude
never tilts.

--mode dynamic: the SAME setpoint is instead handed to OmniDrones' own
``LeePositionController`` (``omni_drones/controllers/lee_position_controller.py``)
each step as ``target_pos``/``target_yaw``, which computes real per-rotor
thrust commands for ``drone.apply_action()`` under real gravity. Chasing a
setpoint that's out ahead of the drone requires real horizontal
acceleration, which the controller achieves by tilting the whole body
toward it (its geometric control law rotates the desired body Z-axis away
from world-Z before computing thrust) -- the same forward-lean a real
quadrotor shows when flying forward, not something this script adds by
hand.

Usage:
    # scene's robot, nothing driven -- just holds station
    python examples/aerial_manipulator_control_demo.py --task AerialBucketGrasp --mode dynamic

    # fly it with the keyboard, arm on the keyboard too
    python examples/aerial_manipulator_control_demo.py --task AerialBucketGrasp --mode dynamic \
        --uav_control keyboard --arm_control keyboard

    # a multi-robot scene: pick which agent you drive. Every other agent in
    # the scene holds the pose it spawned at (without that they get no rotor
    # command at all and drop to the floor).
    python examples/aerial_manipulator_control_demo.py --task AerialAssembly --mode dynamic \
        --uav_control keyboard --arm_control keyboard --agent 1

    # ...and drive the arm from a real SO-ARM leader at the same time
    python examples/aerial_manipulator_control_demo.py --task AerialBucketGrasp --mode dynamic \
        --uav_control keyboard --arm_control soarm_leader --arm-port /dev/ttyACM0

    # no hardware: sine-sweep the arm instead
    python examples/aerial_manipulator_control_demo.py --task AerialBucketGrasp --mode dynamic \
        --arm_control soarm_leader --arm-fake

    # no scene, bare drone + arm on a ground plane
    python examples/aerial_manipulator_control_demo.py --preset iris650_soarm100 \
        --mode dynamic --uav_control keyboard

See docs/TELEOP.md for leader setup (serial permissions, calibration).
"""

import argparse
import csv
import math
import time

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=["kinematic", "dynamic"], default="kinematic")
parser.add_argument("--drone-model", type=str, default="Hummingbird", help="Key into MultirotorBase.REGISTRY.")
parser.add_argument("--task", type=str, default=None,
                    help="Registered scene to fly in (e.g. AerialBucketGrasp). The scene's own "
                         "cfg/task/<name>.yaml declares which robot gets spawned.")
parser.add_argument("--agent", type=int, default=0,
                    help="Which agent of a multi-robot --task scene the control sources drive. "
                         "Every other agent in the scene holds the pose it spawned at.")
parser.add_argument("--preset", type=str, default=None,
                    help="aetos_multiam preset to fly WITH its arm (e.g. iris650_soarm100). "
                         "Takes precedence over --drone-model.")
parser.add_argument("--uav_control", choices=["none", "keyboard"], default="none",
                    help="Where the drone's motion intent comes from. 'none' holds the spawn setpoint.")
parser.add_argument("--arm_control", choices=["none", "keyboard", "soarm_leader"], default="none",
                    help="Where the arm's joint targets come from. 'none' holds the rest pose; "
                         "'keyboard' drives one joint at a time (1-9 select, [/] move, G/H "
                         "gripper) and works on every preset; 'soarm_leader' mirrors a physical "
                         "SO-ARM leader 1:1 and only fits SO-ARM-family manipulators.")
parser.add_argument("--arm-write", choices=["auto", "drive", "teleport"], default="auto",
                    help="How arm joint commands reach the articulation. 'drive' = PD position "
                         "targets (physical, no flicker, and ~600x less drone disturbance where "
                         "the drives work). 'teleport' = write the joint state directly (lag-free "
                         "1:1, the only path that moves so_arm100's shoulder_pan at all, but it "
                         "flickers and kicks the drone). 'auto' (default): teleport for "
                         "soarm_leader, drive for keyboard. See docs/ARM_DISTURBANCE.md.")
parser.add_argument("--arm-joint-speed", type=float, default=0.8,
                    help="rad/s while an --arm_control keyboard joint key is held.")
parser.add_argument("--arm-port", type=str, default="/dev/ttyACM0",
                    help="Serial port of the SOArmLeader (--arm_control soarm_leader).")
parser.add_argument("--arm-fake", action="store_true",
                    help="Sine-sweep the arm instead of reading a real leader -- exercises the "
                         "sim side with no hardware connected.")
parser.add_argument("--arm-fake-joints", type=str, default=None,
                    help="Comma-separated joint names that --arm-fake should actually move "
                         "(default: all of them). e.g. --arm-fake-joints shoulder_pan to isolate "
                         "the one joint whose motion yaws the drone.")
parser.add_argument("--arm-recalibrate", action="store_true",
                    help="Force leader recalibration even if a cached calibration exists.")
parser.add_argument("--arm_dynamics", choices=["frozen", "coupled"], default="frozen",
                    help="'frozen' (default): the arm keeps its static mass/inertia (lumped onto "
                         "base_link) but exerts no reaction on the drone when it moves, so a "
                         "fixed-airframe controller stays stable. 'coupled': real multibody "
                         "physics -- correct, but a hand-tuned Lee controller diverges under "
                         "arm motion. See aetos_multiam/utils/arm_dynamics.py.")
parser.add_argument("--arm-residual", type=float, default=None,
                    help="Fraction of its real mass/inertia each arm link keeps under "
                         "--arm_dynamics frozen. Overrides the scene's own "
                         "task.arm_dynamics_residual; falls back to "
                         "arm_dynamics.DEFAULT_RESIDUAL with no scene. 1.0 == coupled. "
                         "The right value is robot-specific -- iris650_soarm100 wants 0.1 "
                         "and gets WORSE below it, amov950_airbot_bottom wants 0.001.")
parser.add_argument("--arm-lock-joints", type=str, default=None,
                    help="Comma-separated joints held at rest instead of following the "
                         "leader (e.g. shoulder_pan). Overrides the scene's own "
                         "task.arm_locked_joints. Pass an empty string to lock nothing.")
parser.add_argument("--linear-speed", type=float, default=1.0, help="m/s while a move key is held.")
parser.add_argument("--z-speed", type=float, default=0.6, help="m/s while E/Q (ascend/descend) is held.")
parser.add_argument("--yaw-speed", type=float, default=1.2, help="rad/s while a yaw key is held.")
parser.add_argument("--rate", type=float, default=50.0, help="Target control loop rate (Hz).")
parser.add_argument("--log", type=str, default=None,
                    help="Write a per-step CSV here: drone pose/attitude/velocity, the setpoint, "
                         "and the leader's raw + converted + actually-achieved joint values. Use "
                         "it to correlate an arm motion against a drone excursion -- e.g. which "
                         "joint was moving when the body drifted.")
parser.add_argument("--controller", choices=["auto", "lee", "fa"], default="auto",
                    help="auto: 6-DOF allocation on fully-actuated airframes (fahexa750), "
                         "the 4-DOF Lee mixer otherwise. lee/fa force one, for comparison.")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--max_steps", type=int, default=0, help="0 = run until the window closes / Ctrl+C.")
args = parser.parse_args()

# Connect/calibrate the leader BEFORE opening the Isaac Sim window -- see
# this module's docstring, and amteleop_soarm_leader.py's own note.
leader = None
if args.arm_control == "soarm_leader" and not args.arm_fake:
    from aetos_multiam.devices.soarm_leader import SOArmLeader

    leader = SOArmLeader(port=args.arm_port, recalibrate=args.arm_recalibrate)

if args.task:
    import os
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_REPO_ROOT / "cfg")):
        task_cfg = compose(
            config_name="train",
            overrides=[
                f"task={args.task}",
                f"headless={str(args.headless).lower()}",
                # "++" so this is a no-op (not an error) for a scene that has no such
                # key because it never welds its robot in the first place.
                "++task.weld_robot=false",
                # You fly ONE drone with one keyboard; a task's training
                # num_envs (32 for AerialButtonGrasp) just clones it.
                "++task.env.num_envs=1",
            ],
        )
    OmegaConf.resolve(task_cfg)
    OmegaConf.set_struct(task_cfg, False)
    simulation_app = init_simulation_app(task_cfg)
else:
    task_cfg = None
    simulation_app = init_simulation_app({"headless": args.headless})

import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

import omni_drones.utils.scene as scene_utils
from aetos_multiam.controllers import make_position_controller
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler

from aetos_multiam.devices.action_convert import JOINT_ORDER, convert_leader_state_to_radians
from aetos_multiam.devices.keyboard_arm_control import KeyboardArmControl
from aetos_multiam.devices.keyboard_uav_control import KeyboardUAVControl
from aetos_multiam.devices.soarm_leader import MOTOR_LIMITS
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase
from aetos_multiam.utils import freeze_arm_dynamics

#: What read_command() would return with nothing held down -- used in place of
#: polling when --uav_control is "none".
IDLE_UAV_CMD = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}

def _scene_value(key, default):
    """Scene-declared teleop tuning, overridable from the command line. Which
    joints to lock and how hard to freeze the arm are properties of the ROBOT,
    and the scene is what declares the robot -- so they live in
    cfg/task/<name>.yaml next to the preset, not as defaults in here."""
    if task_cfg is None:
        return default
    return task_cfg.task.get(key, default)


#: Joints --arm-fake actually sweeps; the rest are held at their neutral value.
FAKE_JOINTS = (tuple(n.strip() for n in args.arm_fake_joints.split(",") if n.strip())
               if args.arm_fake_joints else JOINT_ORDER)


def fake_joint_state(t: float) -> dict:
    """Sine-sweep each selected joint across ~60% of its leader-normalized
    range, for --arm-fake. Gripper sweeps 0..100 (its own, non-symmetric,
    range). Same sweep amteleop_soarm_leader.py uses, plus --arm-fake-joints
    so one joint can be isolated: only shoulder_pan actually yaws the drone,
    and the rest of the arm's reaction is 2 orders of magnitude smaller."""
    arm_swing = 60.0 * math.sin(t)
    state = {}
    for name in JOINT_ORDER:
        moving = name in FAKE_JOINTS
        if name == "gripper":
            state[name] = 50.0 + (50.0 * math.sin(t * 0.7) if moving else 0.0)
        else:
            state[name] = arm_swing if moving else 0.0
    return state




# Spawn low ("landed") so ascending (E) reads as a real takeoff and
# descending (Q) back toward this height reads as landing -- there's no
# actual ground-contact/motor-spin-down modeling here, just altitude.
# A preset's arm hangs ~0.3m below base_link, so the bare drone's "landed"
# height would bury the gripper in the ground plane. Teleoperating that arm
# needs clearance on top of that -- the arm sweeps through everything under
# the body, reaching ~0.45m down fully extended -- so spawn higher when it
# is being driven. Only --preset is affected; a --task scene places its own
# robot wherever it wants it.
if not args.preset:
    SPAWN_Z = 0.15
elif args.arm_control != "none":
    SPAWN_Z = 1.5
else:
    SPAWN_Z = 0.5
SPAWN_POS = (0.0, 0.0, SPAWN_Z)


class RateLimiter:
    """Sleeps (rendering meanwhile) to hold a target loop rate -- same
    pattern ``amteleop_soarm_leader.py`` uses."""

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


try:
    want_controller = args.mode == "dynamic"
    if args.task:
        from aetos_multiam.environments import make as make_env

        env = make_env(task_cfg.task.name, task_cfg, headless=args.headless)
        env.reset()
        sim = env.sim  # the env built its own SimulationContext; reuse it
        if not 0 <= args.agent < len(env.aerial_manipulators):
            raise SystemExit(
                f"--agent {args.agent} out of range: {task_cfg.task.name} has "
                f"{len(env.aerial_manipulators)} agent(s)."
            )
        am = env.aerial_manipulators[args.agent]  # AerialManipulatorBase, not the bare drone
        drone = am.drone
        controller = (
            make_position_controller(drone, 9.81, args.controller).to(drone.device)
            if want_controller else None
        )
        arm_targets = None  # resolved generically below, once am exists
        # Start from wherever the scene decided to put the robot (env frame --
        # get_state() already subtracts the env origin, and target_pos is fed
        # back into the controller in that same frame).
        _st = drone.get_state()
        SPAWN_POS = tuple(_st[..., :3].reshape(-1).tolist())
        INITIAL_YAW = float(quaternion_to_euler(_st[..., 3:7].reshape(1, 4))[0, 2])
    else:
        # No scene asked for -- plain ground plane, spawn one robot on it.
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=1 / 100,
            rendering_dt=1 / 100,
            backend="torch",
            physics_prim_path="/World/PhysicsScene",
            device="cpu",
            sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, -9.81)},
        )
        scene_utils.design_scene()
        INITIAL_YAW = 0.0

        if args.preset:
            am = AerialManipulatorBase(get_preset(args.preset), name="teleop_am")
            am.spawn(translations=torch.tensor([list(SPAWN_POS)]))
            sim.reset()
            am.initialize()
            drone = am.drone
            # Literal 9.81, not drone.gravity -- see this module's docstring.
            controller = (
                make_position_controller(drone, 9.81, args.controller).to("cpu")
                if want_controller else None
            )
            arm_targets = None  # resolved generically below, once am exists
        else:
            controller_name = "LeePositionController" if want_controller else None
            am = None  # bare drone, no arm at all
            drone, controller = MultirotorBase.make(args.drone_model, controller_name, "cpu")
            drone.spawn(translations=torch.tensor([list(SPAWN_POS)]))
            sim.reset()
            drone.initialize()
            arm_targets = None

    frozen = None
    if args.arm_dynamics == "frozen" and am is not None:
        # Before the controller is used in anger, but after initialize() --
        # it needs the live ArticulationView. Total mass is unchanged (it
        # moves from the arm links onto base_link), so drone.masses and
        # params["mass"], already computed in initialize(), stay correct.
        # CLI > the scene's own yaml > the preset's own measured default.
        _residual = args.arm_residual
        if _residual is None:
            _residual = _scene_value("arm_dynamics_residual", am.cfg.arm_dynamics_residual)
        frozen = freeze_arm_dynamics(am, residual=float(_residual))
        if frozen is not None:
            print(f"[am_control] arm dynamics frozen: {len(frozen['arm_bodies'])} links "
                  f"({frozen['arm_mass']:.3f} kg) lumped onto base_link "
                  f"(now {frozen['base_mass']:.3f} kg, inertia diag "
                  f"{[round(v, 4) for v in frozen['base_inertia_diag']]}, "
                  f"residual={frozen['residual']})", flush=True)

    # --- every other agent in a multi-robot scene -------------------------
    # You drive one of them. The rest would otherwise never get a rotor
    # command and would just fall out of the sky, which makes a two-robot
    # scene pointless to look at -- so each one holds the pose its scene
    # spawned it at, arm parked at rest.
    passive = []
    if args.task:
        for other in env.aerial_manipulators:
            if other is am:
                continue
            if args.arm_dynamics == "frozen":
                _r = args.arm_residual
                if _r is None:
                    _r = other.cfg.arm_dynamics_residual
                freeze_arm_dynamics(other, residual=float(_r))
            _st = other.drone.get_state()
            _idx = other.teleop_joint_indices
            _arm = None
            if _idx is not None:
                _lim = other.drone._view.get_dof_limits()[0, 0][_idx].to(other.drone.device)
                _arm = torch.zeros(1, len(_idx), device=other.drone.device).clamp(
                    _lim[:, 0], _lim[:, 1]
                )
            passive.append({
                "am": other,
                "controller": (
                    make_position_controller(other.drone, 9.81, args.controller)
                    .to(other.drone.device)
                    if want_controller else None
                ),
                "pos": _st[..., :3].reshape(1, 3).clone(),
                "quat": _st[..., 3:7].reshape(1, 4).clone(),
                "yaw": quaternion_to_euler(_st[..., 3:7].reshape(1, 4))[:, 2].clone(),
                "arm": _arm,
            })
        if passive:
            print(f"[am_control] driving agent {args.agent} of "
                  f"{len(env.aerial_manipulators)}; {len(passive)} other agent(s) "
                  f"holding their spawn pose", flush=True)

    # --- the one generic arm handle every teleop device writes through ----
    # Works for presets whose arm is a separate USD welded at spawn time AND
    # for the ones whose arm is baked into the drone's own asset -- the old
    # drone.num_arm_joints/reset_arm_joints path only existed on the latter.
    arm_indices = am.teleop_joint_indices if am is not None else None
    arm_names = am.teleop_joint_names if am is not None else []
    arm_limits = None
    if arm_indices is not None:
        arm_limits = drone._view.get_dof_limits()[0, 0][arm_indices].to(drone.device)
        # Clamp, don't trust: a joint whose range excludes 0 (airbot's joint3
        # is [-0.094, 3.161]) would otherwise start outside its own limit.
        arm_targets = torch.zeros(1, len(arm_indices), device=drone.device).clamp(
            arm_limits[:, 0], arm_limits[:, 1]
        )
        ARM_HOME = arm_targets.clone()
        drone._view.set_joint_positions(
            arm_targets.reshape(1, 1, -1), joint_indices=arm_indices
        )
        drone._view.set_joint_velocities(
            torch.zeros_like(arm_targets).reshape(1, 1, -1), joint_indices=arm_indices
        )
        print(f"[am_control] arm DOFs ({len(arm_names)}): "
              + ", ".join(f"{n}[{float(arm_limits[i, 0]):+.2f},{float(arm_limits[i, 1]):+.2f}]"
                          for i, n in enumerate(arm_names)), flush=True)

    # Which of those DOFs are gripper fingers, and what "open"/"closed" mean
    # for them. A GripperSpec says so outright; a gripper baked into the
    # drone's own USD has no spec, so fall back to each joint's own limits,
    # travelling from the end nearest zero to the far end -- which is what
    # keeps mirrored fingers (amov's endleft [0,+0.04] / endright [-0.04,0])
    # opening in opposite directions instead of both toward +.
    GRIPPER_SLOTS, GRIP_CLOSED, GRIP_OPEN = [], [], []
    if am is not None and arm_indices is not None:
        _spec_grip = {}
        if am.gripper_spec is not None:
            _spec_grip = {
                n: (c, o) for n, c, o in zip(am.gripper_spec.joint_names,
                                             am.gripper_spec.closed_joint_positions,
                                             am.gripper_spec.open_joint_positions)
            }
        _bundled = set(am.drone_spec.bundled_gripper_joint_names)
        for i, n in enumerate(arm_names):
            if n in _spec_grip:
                GRIPPER_SLOTS.append(i)
                GRIP_CLOSED.append(float(_spec_grip[n][0]))
                GRIP_OPEN.append(float(_spec_grip[n][1]))
            elif n in _bundled:
                lo, hi = float(arm_limits[i, 0]), float(arm_limits[i, 1])
                near, far = (lo, hi) if abs(lo) <= abs(hi) else (hi, lo)
                GRIPPER_SLOTS.append(i)
                GRIP_CLOSED.append(near)
                GRIP_OPEN.append(far)
    grip_frac = 0.0

    ARM_WRITE = args.arm_write
    if ARM_WRITE == "auto" and am is not None and am.cfg.arm_write:
        # The preset overrides the per-device default when one of the two write
        # paths is measurably broken on that robot's asset.
        ARM_WRITE = am.cfg.arm_write
    if ARM_WRITE == "auto":
        # teleport is only the right default where the drives can't be
        # trusted (so_arm100's shoulder_pan); everywhere else PD driving is
        # both smoother and far gentler on the drone.
        ARM_WRITE = "teleport" if args.arm_control == "soarm_leader" else "drive"

    def write_arm(targets):
        if ARM_WRITE == "teleport":
            am.teleport_joints(targets)
        else:
            # No joint-velocity write alongside the targets. Tried as a fix for
            # amov950_airbot_dexterous_bottom's diverging Inspire-hand drives
            # and it made them diverge HARDER (1e12 rad vs 1e4), the same trap
            # the teleport path hit -- see the measured table further down.
            am.hold_joints(targets, zero_velocity=False)

    keyboard = KeyboardUAVControl() if args.uav_control == "keyboard" else None
    arm_keyboard = None
    if args.arm_control == "keyboard":
        if arm_indices is None:
            raise ValueError(
                "--arm_control keyboard needs a robot with an arm: use --task or --preset, "
                "not --drone-model."
            )
        arm_keyboard = KeyboardArmControl(arm_names)

    if args.arm_lock_joints is not None:
        LOCKED_JOINTS = tuple(n.strip() for n in args.arm_lock_joints.split(",") if n.strip())
    else:
        _preset_locked = am.cfg.arm_locked_joints if am is not None else ()
        LOCKED_JOINTS = tuple(_scene_value("arm_locked_joints", _preset_locked) or ())

    leader_joint_indices = None
    if args.arm_control == "soarm_leader":
        if am is None or am.arm_joint_indices is None:
            raise ValueError(
                "--arm_control soarm_leader needs a robot with an arm: use --task (a scene "
                "whose own config declares an aerial manipulator) or --preset, not --drone-model."
            )
        manipulator_type = (am.cfg.manipulator_type or "").lower()
        if "so_arm" not in manipulator_type and "soarm" not in manipulator_type:
            raise ValueError(
                f"--arm_control soarm_leader mirrors the leader 1:1 onto SO-ARM joint names "
                f"{JOINT_ORDER}, so it can only drive an SO-ARM-family manipulator -- this "
                f"robot's manipulator_type is {am.cfg.manipulator_type!r}."
            )
        # Arm joints first, gripper last -- exactly the order
        # convert_leader_state_to_radians returns via JOINT_ORDER, which is
        # why teleop_joint_indices is defined that way.
        leader_joint_indices = arm_indices
        if len(leader_joint_indices) != len(JOINT_ORDER):
            raise ValueError(
                f"Leader drives {len(JOINT_ORDER)} joints {JOINT_ORDER} but this robot resolved "
                f"{len(leader_joint_indices)} arm+gripper DOFs -- joint-name lists disagree."
            )

    target_pos = torch.tensor([list(SPAWN_POS)], device=drone.device)  # (1, 3)
    target_yaw = torch.full((1,), INITIAL_YAW, device=drone.device)  # (1,)

    vp = get_active_viewport()
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
    _cx, _cy, _cz = SPAWN_POS
    cam_state.set_position_world(Gf.Vec3d(_cx + 3.0, _cy - 3.0, _cz + 1.5), True)
    cam_state.set_target_world(Gf.Vec3d(_cx, _cy, _cz), True)
except Exception:
    # Kit's sys.excepthook otherwise swallows this silently -- same note as
    # amteleop_soarm_leader.py's own try/except around setup.
    import traceback

    traceback.print_exc()
    simulation_app.close()
    raise

_what = ("task=" + args.task if args.task
         else "preset=" + args.preset if args.preset
         else "drone_model=" + args.drone_model)
_fake = " (fake)" if args.arm_control == "soarm_leader" and args.arm_fake else ""
_locked = f" locked={','.join(LOCKED_JOINTS)}" if LOCKED_JOINTS else ""
_keys = "  -- WASD move, E/Q up/down, Left/Right yaw." if keyboard is not None else ""
_write = f" arm_write={ARM_WRITE}" if arm_indices is not None else ""
print(
    f"[am_control] mode={args.mode} {_what} "
    f"uav_control={args.uav_control} arm_control={args.arm_control}{_fake} "
    f"arm_dynamics={args.arm_dynamics}{_write}{_locked}{_keys}",
    flush=True,
)

log_file = log_writer = None
if args.log:
    log_file = open(args.log, "w", newline="")
    _joint_cols = []
    for n in (JOINT_ORDER if leader_joint_indices is not None else ()):
        _joint_cols += [f"raw_{n}", f"cmd_{n}", f"act_{n}"]
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["step", "sim_t",
         "pos_x", "pos_y", "pos_z",
         "quat_w", "quat_x", "quat_y", "quat_z",
         "roll_deg", "pitch_deg", "yaw_deg", "tilt_deg",
         "vel_x", "vel_y", "vel_z", "angvel_x", "angvel_y", "angvel_z",
         "tgt_x", "tgt_y", "tgt_z", "tgt_yaw",
         "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yawrate"] + _joint_cols
    )
    print(f"[am_control] logging per-step state to {args.log}", flush=True)

rate_limiter = RateLimiter(args.rate)
step = 0
t0 = time.time()
arm_radians = None
try:
    while simulation_app.is_running():
        cmd = keyboard.read_command() if keyboard is not None else IDLE_UAV_CMD
        dt = rate_limiter.sleep_duration

        # Rotate the body-frame FLU intent (vx=forward, vy=left) into world
        # frame by the CURRENT target yaw, so "forward" tracks whichever way
        # the setpoint is already facing, then integrate. Forward is the nose
        # -- the direction the arm reaches -- not 90 deg off it; see
        # devices/keyboard_uav_control.py.
        cos_yaw, sin_yaw = torch.cos(target_yaw), torch.sin(target_yaw)
        dx = (cmd["vx"] * cos_yaw - cmd["vy"] * sin_yaw) * args.linear_speed * dt
        dy = (cmd["vx"] * sin_yaw + cmd["vy"] * cos_yaw) * args.linear_speed * dt
        target_pos[:, 0] += dx
        target_pos[:, 1] += dy
        target_pos[:, 2] += cmd["vz"] * args.z_speed * dt
        target_pos[:, 2].clamp_(min=0.05)
        target_yaw += cmd["yaw_rate"] * args.yaw_speed * dt

        if args.mode == "kinematic":
            rpy = torch.zeros(1, 3)
            rpy[:, 2] = target_yaw
            drone.set_world_poses(target_pos, euler_to_quaternion(rpy))
            drone.set_velocities(torch.zeros(1, 6))
        else:
            drone_state = drone.get_state()[..., :13].squeeze(0)
            action = controller.compute(drone_state, target_pos=target_pos, target_yaw=target_yaw)
            drone.apply_action(action)

        for _p in passive:
            _pam = _p["am"]
            if args.mode == "kinematic" or _p["controller"] is None:
                _pam.drone.set_world_poses(_p["pos"], _p["quat"])
                _pam.drone.set_velocities(torch.zeros(1, 6))
            else:
                _pam.drone.apply_action(_p["controller"].compute(
                    _pam.drone.get_state()[..., :13].squeeze(0),
                    target_pos=_p["pos"], target_yaw=_p["yaw"],
                ))
            if _p["arm"] is not None:
                _pam.hold_joints(_p["arm"], zero_velocity=True)

        if arm_keyboard is not None:
            acmd = arm_keyboard.read_command()
            if acmd["zero_all"]:
                arm_targets = ARM_HOME.clone()
                grip_frac = 0.0
            if arm_names[acmd["active"]] not in LOCKED_JOINTS:
                arm_targets[0, acmd["active"]] += (
                    acmd["joint_rate"] * args.arm_joint_speed * dt
                )
            if GRIPPER_SLOTS and acmd["gripper_rate"]:
                grip_frac = min(1.0, max(0.0, grip_frac + acmd["gripper_rate"] * dt))
                for _k, _slot in enumerate(GRIPPER_SLOTS):
                    arm_targets[0, _slot] = (
                        GRIP_CLOSED[_k] + grip_frac * (GRIP_OPEN[_k] - GRIP_CLOSED[_k])
                    )
            arm_targets.clamp_(arm_limits[:, 0], arm_limits[:, 1])
            arm_radians = arm_targets
            write_arm(arm_targets)
        elif leader_joint_indices is not None:
            joint_state = fake_joint_state(time.time() - t0) if leader is None else leader.read_joint_state()
            motor_limits = MOTOR_LIMITS if leader is None else leader.motor_limits
            radians = convert_leader_state_to_radians(joint_state, motor_limits)
            arm_radians = torch.tensor(
                [[0.0 if n in LOCKED_JOINTS else radians[n] for n in JOINT_ORDER]],
                device=drone.device, dtype=torch.float32,
            )
            # Defaults to a direct position write, not PD targets -- the goal
            # is a lag-free 1:1 leader->sim mapping, and the SO101 asset's
            # shoulder_pan drive outputs no torque at all regardless of gains.
            # Same call (and same reasoning) as amteleop_soarm_leader.py's
            # Teleop.step. --arm-write drive overrides it.
            write_arm(arm_radians)
            # NOT writing a joint velocity alongside this. Doing so removes a
            # visible flicker -- set_joint_positions leaves the solver's own
            # velocity in place, it integrates past the pose just written, and
            # the next teleport snaps it back -- but it costs real attitude
            # quality, and the flicker is the cheaper of the two. Measured over
            # 350 steps of a continuous full-range sweep (per-frame motion the
            # command never asked for / mean drone tilt):
            #
            #   position only            0.144 rad   1.1 deg   <- this
            #   position + velocity 0    0.007 rad  19.9 deg   <- flips the drone
            #   position + d(cmd)/dt     0.007 rad   6.3 deg
            #
            # Zeroing is the trap: it dumps the arm's angular momentum into
            # base_link every step and peaked at 148 deg of tilt, i.e. upside
            # down. (Writing joint position TARGETS changes nothing at all,
            # byte-identical -- this asset's drives produce no torque.)
        elif arm_targets is not None:
            # Every step, in BOTH modes -- the drive instability this prevents
            # has nothing to do with flight dynamics (module docstring).
            am.hold_joints(arm_targets, zero_velocity=True)

        sim.step(render=True)

        if log_writer is not None:
            _s = drone.get_state()
            _p = _s[..., :3].reshape(-1)
            _q = _s[..., 3:7].reshape(-1)
            _v = _s[..., 7:10].reshape(-1)
            _w = _s[..., 10:13].reshape(-1)
            _rpy = quaternion_to_euler(_q.reshape(1, 4)).reshape(-1)
            _upz = float(quat_axis(_q.reshape(1, 4), axis=2).reshape(-1)[2])
            _row = [step, round(step * sim.get_physics_dt(), 4)]
            _row += [round(float(v), 6) for v in _p]
            _row += [round(float(v), 6) for v in _q]
            _row += [round(math.degrees(float(v)), 4) for v in _rpy]
            _row += [round(math.degrees(math.acos(max(-1.0, min(1.0, _upz)))), 4)]
            _row += [round(float(v), 6) for v in _v]
            _row += [round(float(v), 6) for v in _w]
            _row += [round(float(v), 6) for v in target_pos.reshape(-1)]
            _row += [round(float(target_yaw[0]), 6)]
            _row += [round(float(cmd[k]), 4) for k in ("vx", "vy", "vz", "yaw_rate")]
            if leader_joint_indices is not None:
                _act = drone._view.get_joint_positions()[..., leader_joint_indices].reshape(-1)
                _cmd = arm_radians.reshape(-1)
                for _i, _n in enumerate(JOINT_ORDER):
                    _row += [round(float(joint_state[_n]), 4),
                             round(float(_cmd[_i]), 6),
                             round(float(_act[_i]), 6)]
            log_writer.writerow(_row)
            if step % 50 == 0:
                log_file.flush()

        step += 1
        if step % 50 == 0:
            st = drone.get_state()
            pos = [round(v, 3) for v in st[..., :3].reshape(-1).tolist()]
            up_z = float(quat_axis(st[..., 3:7].reshape(1, 4), axis=2)[0, 2])
            arm_note = ""
            if arm_radians is not None:
                # Print what the arm ACTUALLY did next to what was commanded --
                # a write that silently doesn't land (wrong backend, wrong
                # indices) otherwise looks identical to one that works.
                actual = drone._view.get_joint_positions()[..., arm_indices]
                arm_note = (
                    " arm_cmd=" + str([round(v, 2) for v in arm_radians.reshape(-1).tolist()])
                    + " arm_act=" + str([round(v, 2) for v in actual.reshape(-1).tolist()])
                )
            # A held agent that quietly fell on the floor looks exactly like a
            # held agent that is holding, unless its position is printed too.
            held_note = ""
            if passive:
                held_note = " held=" + str([
                    [round(v, 2) for v in _p["am"].drone.get_state()[..., :3].reshape(-1).tolist()]
                    for _p in passive
                ])
            print(
                f"[am_control] step={step} target_pos={target_pos.tolist()} "
                f"target_yaw={round(float(target_yaw[0]), 3)} pos={pos} "
                f"tilt={math.degrees(math.acos(max(-1.0, min(1.0, up_z)))):.1f}deg{arm_note}{held_note}",
                flush=True,
            )
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
    if log_file is not None:
        log_file.close()
    if keyboard is not None:
        keyboard.disconnect()
    if leader is not None:
        leader.disconnect()
    simulation_app.close()
