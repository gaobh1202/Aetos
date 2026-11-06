"""Fly one aerial manipulator through a fixed scripted mission on bare ground.

A stand-in for the real teleop devices that aren't wired up yet (exoskeleton,
Apple Vision Pro, SO-ARM leader): it drives exactly the same setpoint
integration and the same arm write path that
``aerial_manipulator_control_demo.py``'s keyboard does, but from a script, so
every preset can be exercised reproducibly and compared. Use it as the smoke
test for "can this robot fly and move its arm", and as the reference
trajectory when a real device is connected in its place.

The mission, in order:

  idle -> climb to 1 m -> forward 1 m -> left 1 m -> right 1 m -> yaw 180 deg
       -> forward 1 m -> hold while every drivable joint sweeps +45 deg, back,
          -45 deg, back -> land

Translations are body-relative (they follow the current target yaw), so the
"forward" after the 180 deg turn goes back the way the robot came.

The arm sweep is the interesting part: it is the worst case for the drone,
since every joint moves at once. 45 deg is clamped per joint to that joint's
own authored range -- a two-finger gripper's prismatic DOF has a few
centimetres of travel, not 45 degrees -- and the clamped amplitude actually
used is printed per joint, so a joint that could not comply is visible rather
than silently absent.

Per-phase it prints worst body tilt, position error against the setpoint, and
yaw error; the arm-sweep row is the one to read when judging whether
``arm_dynamics_residual`` is low enough for a given robot.

    python examples/show_am_action.py --preset uaquad600_gripper
    python examples/show_am_action.py --preset fahexa750_gripper --headless
    python examples/show_am_action.py --preset all
    python examples/show_am_action.py --preset all --headless --video /tmp/all.mp4
    --preset 'amov950_airbot_bottom',
    'amov950_airbot_dexterous_bottom', 
    'fahexa750_gripper', 
    'hummingbird340_2dof_gripper',
    'iris650_soarm100', 
    'uaquad600_gripper'

Note this duplicates ~20 lines of the demo's setpoint integration rather than
importing it; the demo is a top-level script, not a module, and factoring it
would be a bigger change than the duplication costs.
"""

import argparse
import math
import pathlib
import time

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--preset", type=str, default="uaquad600_gripper",
                    help="aetos_multiam preset to fly (see robots/presets.py), or 'all' to "
                         "spawn every preset side by side in one scene and fly them through "
                         "the mission together.")
parser.add_argument("--arm-residual", type=float, default=None,
                    help="Override the preset's own arm_dynamics_residual. 1.0 == fully "
                         "coupled arm dynamics, which is what the default isolates.")
parser.add_argument("--arm-write", choices=["drive", "teleport"], default=None,
                    help="How the sweep reaches the joints. 'drive' = PD position targets "
                         "(physical, gentle on the drone). 'teleport' = write joint state "
                         "directly (lag-free, flickers, kicks the drone) -- needed where the "
                         "drives are unusable: so_arm100's shoulder_pan, and the Inspire "
                         "hand, whose drives go non-finite under a commanded sweep. Same "
                         "choice as aerial_manipulator_control_demo.py's --arm-write. "
                         "Default: the preset's own arm_write, else 'drive'.")
parser.add_argument("--arm-sweep-deg", type=float, default=45.0,
                    help="Sweep amplitude per joint, clamped to each joint's own range.")
parser.add_argument("--spawn-z", type=float, default=0.5,
                    help="Spawn height. Bottom-mounted arms need the ground clearance; "
                         "the mission climbs to 1 m from wherever this is.")
parser.add_argument("--cruise-z", type=float, default=1.0, help="Height to climb to (m).")
parser.add_argument("--leg", type=float, default=1.0, help="Length of each translation leg (m).")
parser.add_argument("--linear-speed", type=float, default=0.5, help="m/s along a leg.")
parser.add_argument("--z-speed", type=float, default=0.4, help="m/s while climbing or landing.")
parser.add_argument("--yaw-speed", type=float, default=0.8, help="rad/s while turning.")
parser.add_argument("--arm-speed", type=float, default=0.6, help="rad/s per joint while sweeping.")
parser.add_argument("--rate", type=float, default=50.0, help="Control loop rate (Hz).")
parser.add_argument("--video", type=str, default=None,
                    help="Record the mission to this .mp4 (needs ffmpeg on PATH; PNG frames "
                         "are kept next to it in a <stem>_frames/ directory if encoding is "
                         "skipped). Works with --headless. With --preset all the camera frames "
                         "the whole formation instead of chasing one robot.")
parser.add_argument("--capture-every", type=int, default=4,
                    help="Capture one frame every N physics steps. The default pairs with "
                         "--video-fps 25 to play back at real speed (dt is 1/100 s).")
parser.add_argument("--video-fps", type=int, default=25, help="Frame rate of the encoded mp4.")
parser.add_argument("--controller", choices=["auto", "lee", "fa"], default="auto",
                    help="auto: 6-DOF allocation on fully-actuated airframes (fahexa750), "
                         "the 4-DOF Lee mixer otherwise. lee/fa force one, for comparison.")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

simulation_app = init_simulation_app({"headless": args.headless})

import torch  # noqa: E402  (after the app, as every script here does)
from omni.isaac.core.simulation_context import SimulationContext  # noqa: E402

import omni_drones.utils.scene as scene_utils  # noqa: E402
from aetos_multiam.controllers import make_position_controller  # noqa: E402
from omni_drones.utils.torch import quat_axis, quaternion_to_euler  # noqa: E402

from aetos_multiam.robots.presets import PRESETS, get_preset  # noqa: E402
from aetos_multiam.robots.robot_base import AerialManipulatorBase  # noqa: E402
from aetos_multiam.utils import freeze_arm_dynamics  # noqa: E402

DT = 1.0 / 100.0


class Recorder:
    """Capture the viewport to PNGs each Nth step, then encode an mp4.

    Rendering is what costs here, so it only happens on a captured step --
    which is also why this works under --headless: the viewport still renders
    on demand, it just is not shown.
    """

    def __init__(self, path, rigs):
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        self.path = pathlib.Path(path)
        self.frames_dir = self.path.with_name(self.path.stem + "_frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.frames_dir.glob("frame_*.png"):
            stale.unlink()
        self.viewport = get_active_viewport()
        self.camera = ViewportCameraState("/OmniverseKit_Persp", self.viewport)
        self.rigs = rigs
        self.index = 0
        # One robot: chase it from behind and above. Several: back off far
        # enough to hold the whole line of them, which is why the offset
        # scales with how far apart they are spawned.
        if len(rigs) == 1:
            # Scale the chase distance with the airframe, so a 340 mm
            # hummingbird and a 950 mm amov fill the frame about equally --
            # at a fixed 3 m the small one is a speck next to the big one.
            scale = (rigs[0].am.drone_spec.class_size_mm or 650) / 650.0
            self.offset = (-2.0 * scale, -2.2 * scale, 1.1 * scale)
        else:
            span = ROBOT_SPACING_M * (len(rigs) - 1)
            self.offset = (0.0, -(span * 0.9 + 6.0), span * 0.35 + 3.0)

    def capture(self, step):
        from omni.kit.viewport.utility import capture_viewport_to_file
        from pxr import Gf

        if step % args.capture_every:
            return
        centre = torch.stack([
            rig.drone.get_state()[..., :3].reshape(-1) for rig in self.rigs
        ]).mean(0)
        cx, cy, cz = (float(v) for v in centre)
        ox, oy, oz = self.offset
        self.camera.set_position_world(Gf.Vec3d(cx + ox, cy + oy, cz + oz), True)
        self.camera.set_target_world(Gf.Vec3d(cx, cy, cz), True)
        sim = SimulationContext.instance()
        sim.render()
        capture_viewport_to_file(self.viewport,
                                 str(self.frames_dir / f"frame_{self.index:05d}.png"))
        self.index += 1

    def encode(self):
        import shutil
        import subprocess

        if self.index == 0:
            print("[show_am] no frames captured, nothing to encode", flush=True)
            return
        # Prefer imageio_ffmpeg's bundled binary: the system ffmpeg on this
        # machine is built without libx264, and mpeg4 is a visibly worse
        # fallback that some browsers will not play.
        ffmpeg = None
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print(f"[show_am] no ffmpeg available; {self.index} frames left in "
                  f"{self.frames_dir}", flush=True)
            return

        def run(codec, extra):
            # -vf pad: the capture resolution can be odd-numbered, which both
            # libx264 and mpeg4 reject.
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.video_fps),
                   "-i", str(self.frames_dir / "frame_%05d.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", codec, *extra, str(self.path)]
            return subprocess.run(cmd, capture_output=True, text=True)

        result = run("libx264", ["-pix_fmt", "yuv420p", "-crf", "20"])
        codec = "libx264"
        if result.returncode:
            result = run("mpeg4", ["-q:v", "3"])
            codec = "mpeg4"
        if result.returncode:
            print(f"[show_am] ffmpeg failed ({result.returncode}): "
                  f"{result.stderr.strip()[:400]}", flush=True)
            print(f"[show_am] {self.index} frames kept in {self.frames_dir}", flush=True)
            return
        size_mb = self.path.stat().st_size / 1e6
        print(f"[show_am] wrote {self.path} ({self.index} frames, {codec}, "
              f"{args.video_fps} fps, {size_mb:.1f} MB)", flush=True)
        shutil.rmtree(self.frames_dir, ignore_errors=True)


ROBOT_SPACING_M = 4.0  # the mission's legs are 1 m; 4 m never lets two robots meet


class Rig:
    """One aerial manipulator running the mission: its own controller, its own
    setpoint, its own arm. Several of them share one SimulationContext, which
    is the whole point of ``--preset all``."""

    def __init__(self, preset_name, instance, x_offset):
        self.preset = preset_name
        self.tag = preset_name.replace("_gripper", "").replace("_bottom", "")[:16]
        self.am = AerialManipulatorBase(get_preset(preset_name), name=f"show_am_{instance}")
        self.am.spawn(translations=torch.tensor([[x_offset, 0.0, args.spawn_z]]))

    def attach(self):
        """Create this robot's ArticulationView. Separate from ``configure``
        because every view has to exist before any of them is written to: Isaac
        hands each view a physics-backend buffer, and creating a later view
        invalidates the earlier ones ("Failed to get DOF position targets from
        backend" on the first write, with six robots in the scene)."""
        self.am.initialize(reset_rest_pose=False)
        self.drone = self.am.drone
        self.controller = make_position_controller(
            self.drone, 9.81, args.controller).to(self.drone.device)
        print(f"[show_am] {self.tag}: {type(self.controller).__name__}", flush=True)

    def configure(self):
        am = self.am
        drone = self.drone
        residual = args.arm_residual
        if residual is None:
            residual = am.cfg.arm_dynamics_residual
        frozen = freeze_arm_dynamics(am, residual=float(residual))
        self.arm_write = args.arm_write or am.cfg.arm_write or "drive"
        print(f"[show_am] {self.tag}: residual={residual} arm_write={self.arm_write}"
              + (f" ({len(frozen['arm_bodies'])} links, {frozen['arm_mass']:.3f} kg onto "
                 f"base_link -> {frozen['base_mass']:.3f} kg)" if frozen else " (no arm)"),
              flush=True)

        self.indices = am.teleop_joint_indices
        self.names = am.teleop_joint_names
        if self.indices is None:
            raise SystemExit(f"{self.preset} has no arm; this mission has nothing to sweep.")
        self.limits = drone._view.get_dof_limits()[0, 0][self.indices].to(drone.device)
        self.home = torch.zeros(1, len(self.indices), device=drone.device).clamp(
            self.limits[:, 0], self.limits[:, 1])
        # 45 deg each way, or as much of it as each joint's range allows --
        # clamped per side, not symmetrically. A two-finger gripper's prismatic
        # DOF has centimetres of travel starting at zero, so a symmetric clamp
        # would pin it to 0 and it would never move at all; this way it travels
        # as far as it can in the direction it has room for.
        want = math.radians(args.arm_sweep_deg)
        self.amp_pos = torch.minimum(torch.full_like(self.home[0], want),
                                     self.limits[:, 1] - self.home[0]).clamp(min=0.0)
        self.amp_neg = torch.minimum(torch.full_like(self.home[0], want),
                                     self.home[0] - self.limits[:, 0]).clamp(min=0.0)
        print(f"[show_am] {self.tag}: sweep (+/-) "
              + ", ".join(f"{n}=+{math.degrees(float(self.amp_pos[i])):.0f}/"
                          f"-{math.degrees(float(self.amp_neg[i])):.0f}"
                          for i, n in enumerate(self.names)) + " deg", flush=True)
        locked = set(am.cfg.arm_locked_joints)
        if locked:
            for i, n in enumerate(self.names):
                if n in locked:
                    self.amp_pos[i] = self.amp_neg[i] = 0.0
            print(f"[show_am] {self.tag}: held at rest: {sorted(locked)}", flush=True)

        drone._view.set_joint_positions(self.home.reshape(1, 1, -1), joint_indices=self.indices)
        drone._view.set_joint_velocities(torch.zeros_like(self.home).reshape(1, 1, -1),
                                         joint_indices=self.indices)

        state = drone.get_state()
        self.target_pos = state[..., :3].reshape(1, 3).clone()
        self.target_yaw = quaternion_to_euler(state[..., 3:7].reshape(1, 4))[:, 2].clone()
        self.arm_targets = self.home.clone()
        self.arm_frac = torch.zeros((), device=drone.device)  # -1..+1 of amp_pos / amp_neg
        self.touchdown_at = None
        self.landing = False
        self.grind_heights = []
        # Peak travel each way during the sweep phases. Without this, a joint
        # that never moved is indistinguishable from one that moved and came
        # back -- exactly how hummingbird's undriven R1/R2 went unnoticed.
        self.peak_lo = self.home[0].clone()
        self.peak_hi = self.home[0].clone()
        self.worst_tilt = 0.0
        self.tilt_max = 0.0

    def begin_phase(self, vz):
        self.tilt_max = 0.0
        self.landing = self.landing or vz < 0

    def control(self, vx, vy, vz, yaw_rate, arm_dir):
        want = math.radians(args.arm_sweep_deg)
        # Needs BOTH: a landing has been commanded at some point (the flag
        # persists past the descent phase, because it is the next phase --
        # holding a setpoint underground -- that grinds the robot into the
        # floor), and the setpoint is now below the body. Keying on "below the
        # body and not moving" alone false-fires during the climb, when an
        # overshoot briefly puts the body above the ramp: measured, it froze
        # the setpoint at 1.02 m and the robot then never landed at all.
        self.grinding = (self.landing
                         and float(self.target_pos[0, 2])
                         < float(self.drone.get_state()[..., 2]) - 0.02)
        if not self.grinding:
            self.grind_heights = []
        elif self.touchdown_at is None:
            # 0.3 s of commanded-but-refused descent, within 5 mm: it is
            # resting on its legs, and the setpoint should stop chasing the
            # ground. The window starts when grinding starts, deliberately:
            # measured against the whole phase instead, it fired in the lag
            # window at the top of the descent (the body has not started moving
            # yet, so it looks perfectly stalled) and froze the setpoint at
            # cruise height. 5 mm rather than 1 mm because a robot bouncing on
            # its legs never settles to 1 mm, and uaquad600 then ground into
            # the floor for the whole last phase.
            if (len(self.grind_heights) >= 30
                    and max(self.grind_heights) - min(self.grind_heights) < 5e-3):
                self.touchdown_at = float(self.grind_heights[-1])
                self.target_pos[:, 2] = self.touchdown_at
                print(f"[show_am] {self.tag}: touchdown at z={self.touchdown_at:.3f}m", flush=True)

        cos_yaw, sin_yaw = torch.cos(self.target_yaw), torch.sin(self.target_yaw)
        self.target_pos[:, 0] += (vx * cos_yaw - vy * sin_yaw) * args.linear_speed * DT
        self.target_pos[:, 1] += (vx * sin_yaw + vy * cos_yaw) * args.linear_speed * DT
        if self.touchdown_at is None:
            self.target_pos[:, 2] += vz * args.z_speed * DT
            self.target_pos[:, 2].clamp_(min=0.05)
        self.target_yaw += yaw_rate * args.yaw_speed * DT

        if arm_dir is not None:
            self.arm_frac = (self.arm_frac
                             + arm_dir * args.arm_speed * DT / max(want, 1e-9)).clamp(-1, 1)
            side = self.amp_pos if float(self.arm_frac) >= 0.0 else self.amp_neg
            self.arm_targets = (self.home + self.arm_frac * side).clamp(
                self.limits[:, 0], self.limits[:, 1])
        if self.arm_write == "teleport":
            self.am.teleport_joints(self.arm_targets)
        else:
            self.am.hold_joints(self.arm_targets)

        drone_state = self.drone.get_state()[..., :13].squeeze(0)
        self.drone.apply_action(self.controller.compute(
            drone_state, target_pos=self.target_pos, target_yaw=self.target_yaw))

    def observe(self, arm_dir):
        measured = self.drone.get_state()
        up_z = float(quat_axis(measured[..., 3:7].reshape(1, 4), axis=2).reshape(-1)[2])
        self.tilt_max = max(self.tilt_max, math.degrees(math.acos(max(-1.0, min(1.0, up_z)))))
        if arm_dir is not None:
            achieved = self.drone._view.get_joint_positions()[..., self.indices].reshape(-1)
            self.peak_lo = torch.minimum(self.peak_lo, achieved)
            self.peak_hi = torch.maximum(self.peak_hi, achieved)
        if self.grinding:
            self.grind_heights.append(float(measured[..., 2]))
            if len(self.grind_heights) > 30:
                self.grind_heights.pop(0)

    def phase_row(self, label):
        measured = self.drone.get_state()
        pos = measured[..., :3].reshape(-1)
        if not torch.isfinite(pos).all():
            raise SystemExit(
                f"[show_am] FAILED: {self.preset} went non-finite during '{label}'")
        pos_err = float((pos - self.target_pos.reshape(-1)).norm())
        # Wrap into (-180, 180]: raw subtraction of two angles reports a 2 deg
        # error as 358 deg, which reads like total loss of yaw control.
        yaw_err = math.degrees(abs((float(
            quaternion_to_euler(measured[..., 3:7].reshape(1, 4))[0, 2] - self.target_yaw[0]
        ) + math.pi) % (2 * math.pi) - math.pi))
        self.worst_tilt = max(self.worst_tilt, self.tilt_max)
        return (f"[show_am] {label:10s} {self.tag:17s} {self.tilt_max:7.2f}d "
                f"{pos_err:7.3f}m {yaw_err:7.2f}d {float(pos[2]):6.3f}m")

    def summary(self):
        lines = ["[show_am] " + self.tag + ": travel reached (min..max during the sweep) "
                 + ", ".join(f"{n}={math.degrees(float(self.peak_lo[i])):+.0f}.."
                             f"{math.degrees(float(self.peak_hi[i])):+.0f}"
                             for i, n in enumerate(self.names)) + " deg"]
        pos = self.drone.get_state()[..., :3].reshape(-1)
        lines.append(f"[show_am] {self.tag}: worst body tilt {self.worst_tilt:.2f} deg, "
                     f"landed at z={float(pos[2]):.3f} m")
        return lines


def main():
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=DT,
        rendering_dt=DT,
        backend="torch",
        physics_prim_path="/World/PhysicsScene",
        device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, -9.81)},
    )
    scene_utils.design_scene()

    preset_names = list(PRESETS) if args.preset == "all" else [args.preset]
    rigs = [Rig(name, i, i * ROBOT_SPACING_M) for i, name in enumerate(preset_names)]
    sim.reset()
    for rig in rigs:
        rig.attach()
    # One reset for the whole scene, then rebind everything. A mount_child'd
    # arm needs a second rest-pose solve (see AerialManipulatorBase.initialize),
    # but that reset is global: letting each robot do its own would leave the
    # others holding stale view handles, which segfault on the next read.
    sim.reset()
    for rig in rigs:
        rig.am.reattach()
    for rig in rigs:
        rig.configure()

    def seconds(n):
        return int(round(n / DT))

    # (label, steps, (vx, vy, vz, yaw_rate), arm_phase)
    # vx = right, vy = forward, both in the current target-yaw frame.
    leg = seconds(args.leg / args.linear_speed)
    climb = seconds((args.cruise_z - args.spawn_z) / args.z_speed)
    turn = seconds(math.pi / args.yaw_speed)
    sweep = seconds(math.radians(args.arm_sweep_deg) / args.arm_speed)
    mission = [
        ("idle",      seconds(1.0),  (0, 0, 0, 0),   None),
        ("climb",     climb,         (0, 0, 1, 0),   None),
        ("settle",    seconds(1.5),  (0, 0, 0, 0),   None),
        ("forward",   leg,           (0, 1, 0, 0),   None),
        ("settle",    seconds(1.0),  (0, 0, 0, 0),   None),
        ("left",      leg,           (-1, 0, 0, 0),  None),
        ("right",     leg,           (1, 0, 0, 0),   None),
        ("settle",    seconds(1.0),  (0, 0, 0, 0),   None),
        ("yaw180",    turn,          (0, 0, 0, 1),   None),
        ("settle",    seconds(1.0),  (0, 0, 0, 0),   None),
        ("forward",   leg,           (0, 1, 0, 0),   None),
        ("settle",    seconds(1.5),  (0, 0, 0, 0),   None),
        ("arm +45",   sweep,         (0, 0, 0, 0),   +1),
        ("arm back",  sweep,         (0, 0, 0, 0),   -1),
        ("arm -45",   sweep,         (0, 0, 0, 0),   -1),
        ("arm back",  sweep,         (0, 0, 0, 0),   +1),
        ("settle",    seconds(1.0),  (0, 0, 0, 0),   None),
        ("land",      seconds((args.cruise_z - 0.05) / args.z_speed), (0, 0, -1, 0), None),
        ("idle",      seconds(1.0),  (0, 0, 0, 0),   None),
    ]

    recorder = Recorder(args.video, rigs) if args.video else None
    step = 0
    # No pacing while recording: every captured step already renders, and
    # sleeping on top of that only makes a long mission longer.
    sleep_per_step = 0.0 if (args.headless or recorder) else 1.0 / args.rate
    print(f"[show_am] {'phase':10s} {'robot':17s} {'tilt_max':>8s} {'pos_err':>8s} "
          f"{'yaw_err':>8s} {'height':>7s}", flush=True)

    for label, steps, (vx, vy, vz, yaw_rate), arm_dir in mission:
        for rig in rigs:
            rig.begin_phase(vz)
        for _ in range(steps):
            for rig in rigs:
                rig.control(vx, vy, vz, yaw_rate, arm_dir)
            sim.step(render=not args.headless)
            if sleep_per_step:
                time.sleep(sleep_per_step)
            for rig in rigs:
                rig.observe(arm_dir)
            if recorder is not None:
                recorder.capture(step)
            step += 1
        for rig in rigs:
            print(rig.phase_row(label), flush=True)

    for rig in rigs:
        for line in rig.summary():
            print(line, flush=True)
    print(f"[show_am] DONE {len(rigs)} robot(s), {step} steps", flush=True)
    if recorder is not None:
        recorder.encode()


try:
    main()
except SystemExit as exc:
    print(exc, flush=True)
    raise
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
