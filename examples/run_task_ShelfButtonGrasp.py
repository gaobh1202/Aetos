"""Run the ShelfButtonGrasp scene as an actual task: scene + a real control
source driving the drone and a real control source driving the arm,
independently of each other -- this is the piece raids-am-simulator had
that the plain scene migration (environments/single/shelf_button_grasp.py)
didn't: a runnable combination of "what flies the drone" and "what moves
the arm", matching what raids-am-simulator could already do:

  1. drone flown by a trained RL policy (raids used a hand-tuned PID instead
     -- OmniDrones has an actual trained hover policy, reused here).
  2. arm (+ the 4 shelf props) driven by a pre-recorded trajectory, exactly
     like raids-am-simulator's own replay_isaacsim.py.

OmniDrones already has a script doing exactly this combination:
``/home/benhua/OmniDrones/scripts/run_shelf_grasp.py`` (built on the same
``hover.py`` task class's ``arm_hold``/``arm_replay`` machinery). This
script's FLY_IN/REPLAY phase machine, playback-speed remapping, and
continuous arm-velocity damping are all ported from it -- read it first if
you're debugging this one.

ROBOT ASSET -- why this doesn't use aetos_multiam's own composable preset
system: this script spawns ``amov_drone_arm_1.usd`` directly (OmniDrones'
own pre-baked merged drone+arm USD, via OmniDrones' own ``AmovDrone`` class
-- see AMOV_DRONE_ARM_USD_PATH below) instead of using
``AerialManipulatorBase`` + the ``amov950_airbot_bottom`` preset (drone +
a separately-vendored copy of the airbot_play arm, mounted at runtime via
``mount_child``). Those two are the *same physical robot* -- same drone,
same arm, same mount geometry (verified to 4 decimal places) -- but not
the same asset *file*: the vendored copy's ``arm_base`` body turned out to
be 0.777 kg against the training asset's 0.166 kg (confirmed by dumping
both files' UsdPhysics.MassAPI values; every other body matched exactly).
Correcting just the mass at runtime didn't change the simulated dynamics
at all -- PhysX's cached inertia tensor, computed once at parse time from
density + the (wrong) original mass, isn't recomputed by a bare
``set_body_masses()`` call, so mass and inertia silently went out of sync
with each other instead of actually matching the training asset. Rather
than chase every other property PhysX might cache this way, this script
just spawns the literal bit-for-bit file the checkpoint was trained
against -- aetos_multiam's own preset/composition system exists for
*swappable* drone x arm x gripper combos, which isn't the goal here at
all; the goal is reproducing one specific checkpoint's training setup
exactly, and the merged USD does that for free.

Drone control source (--drone-control):
  policy (default) -- a trained RL hover policy keeps the drone steady at
    its spawn pose. See DronePolicyController: it replicates OmniDrones'
    own Hover env observation formula by hand (omni_drones/envs/single/
    hover.py:678-707, rpos + drone_state[3:] + rheading + time_encoding)
    since the checkpoint's actor network was trained on exactly that
    layout (task=HoverAmovDynaArmReplay; see that run's own config.yaml for
    the confirmed hyperparameters this script matches).
  device -- NOT YET IMPLEMENTED. No drone-control input device exists in
    this project yet (e.g. a second joystick/leader channel) -- extension
    point, see DroneDeviceController.

Arm control source (--arm-control):
  replay (default) -- replays one recorded raids-am-simulator episode (arm
    joint angles + gripper + all 4 object poses) via PD position targets.
    See ArmReplayController; data migrated from raids-am-simulator/
    raids_am_Scene/replay/out/episodes_npz/train_button_grasp/ into
    aetos_multiam/data/shelf_button_grasp/episodes/ -- the same episode set
    OmniDrones' own HoverAmovDynaArmReplay task trains against.
  policy -- NOT YET IMPLEMENTED. No manipulation policy has been trained
    for this robot's action/observation layout yet -- extension point,
    see ArmPolicyController.
  device -- NOT YET IMPLEMENTED. Would need an airbot_play-compatible
    physical leader; none exists yet (examples/amteleop_soarm_leader.py's
    SOArmLeader is SO-ARM-family only) -- extension point, see
    ArmDeviceController.

This took several real fixes to get stable, worth knowing if you touch
this again:

  1. The 4 replayed props' kinematic flag must be authored BEFORE the
     first sim.reset() -- flipping it afterward doesn't reach the GPU
     PhysX pipeline's already-compiled solver, so the body stays
     effectively dynamic and every replayed teleport reads as a violent
     collision (constant jitter, objects launched away).
  2. Any internal joint within a prop (ShelfButton's 2 decorative
     sub-button PrismaticJoints, PowerDrill's trigger joint) has to be
     deactivated before going kinematic -- PhysX refuses to create a
     joint whose both ends are about to be kinematic, aborting the whole
     stage parse. See ``aetos_multiam.utils.make_kinematic`` (handles this
     generically) and ``ShelfButton`` (also strips its ArticulationRootAPI
     -- E_body_16 is actually an articulation root). The two sub-buttons
     consequently stay frozen wherever they spawn -- a cosmetic gap only.
  3. The arm's joint POSITION (not just its PD target) has to be
     teleported to the replay's starting frame, with its velocity zeroed,
     BEFORE the very first physics step -- same three calls (position,
     velocity, target) ``AmovDrone.reset_arm_joints`` makes (itself a port
     of ``hover.py``'s own arm_replay ``_reset_idx``). Skipping this leaves
     the arm at its USD rest pose while the PD target jumps straight to the
     recorded frame 0 on step 0; the resulting near-instantaneous PD
     correction's reaction torque, transmitted through the arm mount, was
     enough to flip the drone on its own.
  4. The recorded trajectory (30 fps) must be remapped onto the sim's own
     control rate (1/physics_dt) instead of indexed 1:1 by control step --
     doing it 1:1 replayed the arm roughly 2x too fast, doubling every
     joint velocity/PD-correction the drone had to absorb.
     ``run_shelf_grasp.py`` fixes this two ways, both ported here: (a)
     remap elapsed control steps to a frame index via
     ``elapsed * (dataset_fps / control_hz)`` (see the FLY_IN/REPLAY loop
     below); (b) hold the arm fixed at frame 0 (velocity zeroed every step
     -- ``AmovDrone.hold_arm_joints(zero_velocity=True)``) while the hover
     policy flies in from a slightly offset spawn and settles, and only
     start the replay's frame index advancing once position error and
     speed both stay under threshold for N consecutive steps (or a
     timeout) -- so the arm never starts moving until the policy is ready.
  5. The arm's PD gains must match what HoverAmovDynaArmReplay actually
     trained with (task.arm_replay.stiffness=500, damping=50 -- confirmed
     in run-20260922_084532-yau99ypc/files/config.yaml), applied to the
     same combined arm+gripper joint set ``hold_arm_joints`` drives.
  6. The policy's raw [-1,1] action must be run through the same affine
     remap ``hover.py``'s own ``_pre_sim_step`` applies before ever calling
     ``apply_action`` (``hover.py:589-624``): ``action*hover_cmd_scale +
     hover_cmd_offset``, so that action=0 lands exactly on hover thrust
     instead of RotorGroup's own raw cmd=0. The training policy was always
     evaluated through this remap -- skipping it here is a silent,
     systematic thrust bias that looked like a steady-state position
     offset during FLY_IN (pos_err plateaued ~0.3m instead of settling)
     and, worse, left materially less disturbance margin once REPLAY
     actually started moving the arm. ``hover_frac``/``offset``/``scale``
     are computed once from ``drone.gravity``/``drone.KF`` right after
     ``initialize()`` and printed at startup -- confirmed to match
     ``scripts/run_shelf_grasp.py``'s own logged values bit-for-bit
     (hover_frac=0.2494) once asset fix (ROBOT ASSET section) was also in
     place.

Verified in Isaac Sim (headless, GPU PhysX) against OmniDrones' own
scripts/run_shelf_grasp.py as a baseline, using the same checkpoint and
the same ep000.npz episode: both the baseline and this script now
complete the full 605-frame replay without a flip, each recovering from a
large mid-episode disturbance spike (pos_err to ~1.2-1.8, up_z dipping to
~0.93-0.96 but never approaching the 0.5 flip threshold) rather than
diverging. Getting here took both ROBOT ASSET (spawn the exact training
USD) and fix 6 (the hover action remap) together -- the asset fix alone
(still missing fix 6) still flipped, just later (~95% through instead of
~80%), confirming the systematic thrust bias was eating into the policy's
disturbance margin independently of the asset mismatch.

Usage:
    python examples/run_task_ShelfButtonGrasp.py
    python examples/run_task_ShelfButtonGrasp.py --arm-control replay --drone-control policy
    python examples/run_task_ShelfButtonGrasp.py --headless --max_steps 1200
"""

import os

# Must be set before omni_drones.robots.drone.amov_drone is ever imported
# (even transitively) -- AmovDrone's usd_path/param_path class attributes
# are resolved from this env var once, at class-definition time.
AMOV_DRONE_ARM_USD_PATH = "/home/benhua/OmniDrones/amov_drone_arm_1.usd"
os.environ.setdefault("OMNIDRONES_AMOV_USD_PATH", AMOV_DRONE_ARM_USD_PATH)

import argparse
from pathlib import Path

import numpy as np
import torch

from aetos_multiam import init_simulation_app

# Hand-tuned scene layout read back from raids-am-simulator's shelf_grasp.usd
# (raids_am_Scene/build_isaacsim_scene.py) -- same constants as
# aetos_multiam.environments.single.shelf_button_grasp.ShelfButtonGrasp.
GEARWHEEL_POS = (-0.006728, 0.038057, 1.386416)
BUTTON_POS = (0.172601, 0.045351, 1.365782)
DRILL_POS = (0.376657, 0.021583, 1.441908)
DRILL_YAW_DEG = 180.0
TOTE_BOX_POS = (-0.276033, 0.009358, 1.345964)
TOTE_BOX_YAW_DEG = 90.0
ROBOT_POS = (-0.099038, 0.438106, 2.002114)
ROBOT_YAW_DEG = -90.0

# From the sibling OmniDrones repo's own trained run (task=HoverAmovDynaArmReplay,
# continuing run-20260601_002626-amov_drone_dynamic_arm_hover's checkpoint;
# algo.priv_actor=false, algo.priv_critic=false, algo.actor_log_std_init=-1.0,
# env.max_episode_length=1500 -- see that run's own config.yaml). This is the
# same Hover-family observation/reward formula as plain HoverAmov (both
# subclass hover.py's Hover), just continued-trained with the arm driven by
# real recorded episodes instead of held fixed -- matches this script's own
# arm-control=replay default. Not vendored into aetos_multiam: it's a live
# training artifact in a repo this project already depends on.
DEFAULT_DRONE_POLICY_CKPT = (
    "/home/benhua/OmniDrones/wandb/run-20260922_084532-yau99ypc/files/checkpoint_latest.pt"
)
DEFAULT_DRONE_POLICY_MAX_EPISODE_LENGTH = 1500
_EPISODES_DIR = Path(__file__).resolve().parent.parent / "aetos_multiam" / "data" / "shelf_button_grasp" / "episodes"
DEFAULT_EPISODE_PATHS = sorted(str(p) for p in _EPISODES_DIR.glob("*.npz"))

OBJECT_NAMES = ("gearwheel", "button", "power_drill", "tote_box")

# arm_replay's own gains (see fix 5 in the module docstring).
ARM_GAIN_STIFFNESS = 500.0
ARM_GAIN_DAMPING = 50.0


# ---------------------------------------------------------------------------
# Drone control sources
# ---------------------------------------------------------------------------

class DroneController:
    def get_action(self, drone, step: int) -> torch.Tensor:
        """Return a (1, num_rotors) action in the drone's own [-1, 1] convention."""
        raise NotImplementedError


class DronePolicyController(DroneController):
    """Trained RL hover policy -- holds the drone at its spawn pose/heading.

    Builds the actor/critic lazily on the first call, once ``drone`` (and so
    ``drone.num_rotors``/``device``) actually exists.
    """

    def __init__(self, checkpoint_path: str, target_pos, target_yaw_deg: float,
                 max_episode_length: int = DEFAULT_DRONE_POLICY_MAX_EPISODE_LENGTH,
                 time_encoding: bool = True):
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"--drone-policy-ckpt {checkpoint_path!r} does not exist. Train one with "
                f"OmniDrones (task=HoverAmovDynaArmReplay algo=ppo) or pass a different "
                f"--drone-policy-ckpt."
            )
        self.checkpoint_path = checkpoint_path
        self.target_pos_list = list(target_pos)
        self.target_yaw_deg = target_yaw_deg
        self.max_episode_length = max_episode_length
        self.time_encoding = time_encoding
        self.time_encoding_dim = 4 if time_encoding else 0
        self._policy = None

    def _build(self, drone) -> None:
        from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

        from omni_drones.learning.ppo import PPOPolicy
        from omni_drones.learning.ppo.ppo import PPOConfig
        from omni_drones.utils.torch import quat_axis

        device = drone.device
        num_rotors = drone.num_rotors
        obs_dim = 3 + (16 + num_rotors) + 3 + self.time_encoding_dim

        # observation_spec stays a full {"agents": {"observation": ...}} composite --
        # PPOPolicy's fake_input = observation_spec.zero() needs that nested
        # structure for its TensorDictModule to read [("agents","observation")]
        # back out. action_spec/reward_spec, by contrast, must be the BARE leaf
        # spec: that's what env.action_spec/env.reward_spec actually hand
        # PPOPolicy in real training (torchrl's EnvBase unwraps a composite with
        # exactly one action/reward key down to that key's own leaf spec), and
        # PPOPolicy reads their .shape directly (not through a key).
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, 1, obs_dim), device=device),
            }),
        })
        action_spec = BoundedTensorSpec(-1, 1, (1, 1, num_rotors), device=device)
        reward_spec = UnboundedContinuousTensorSpec((1, 1, 1), device=device)

        cfg = PPOConfig(checkpoint_path=self.checkpoint_path, actor_log_std_init=-1.0)
        self._policy = PPOPolicy(cfg, observation_spec, action_spec, reward_spec, device=device)

        self.target_pos = torch.tensor([[self.target_pos_list]], device=device)  # (1, 1, 3)
        yaw = torch.tensor(self.target_yaw_deg * 3.141592653589793 / 180.0, device=device)
        target_quat = torch.stack([
            torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2),
        ]).unsqueeze(0)
        self.target_heading = quat_axis(target_quat, 0).reshape(1, 1, 3)

    def get_action(self, drone, step: int) -> torch.Tensor:
        from tensordict import TensorDict

        if self._policy is None:
            self._build(drone)

        drone_state = drone.get_state()  # (1, 1, 19 + num_rotors)
        safe = torch.nan_to_num(drone_state, nan=0.0, posinf=0.0, neginf=0.0)

        # Same formula as omni_drones/envs/single/hover.py:685-688.
        rpos = self.target_pos - safe[..., :3]
        rheading = self.target_heading - safe[..., 13:16]
        obs = [rpos, safe[..., 3:], rheading]
        if self.time_encoding:
            t = torch.full((1, 1, 1), min(step / self.max_episode_length, 1.0), device=safe.device)
            obs.append(t.expand(1, 1, self.time_encoding_dim))
        obs = torch.cat(obs, dim=-1)

        td = TensorDict({"agents": {"observation": obs}}, batch_size=[1])
        with torch.no_grad():
            self._policy(td)
        return td[("agents", "action")]


class DroneDeviceController(DroneController):
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "--drone-control=device: no drone-control input device exists in this project yet "
            "(e.g. a joystick, or a second leader channel). Extension point -- see "
            "examples/amteleop_soarm_leader.py's SOArmLeader for the device-teleop pattern this "
            "would follow once a drone-control device exists."
        )


# ---------------------------------------------------------------------------
# Arm (+ object) control sources
# ---------------------------------------------------------------------------

class ArmController:
    def get_targets(self, frame_idx: int):
        """Return (arm_joint_targets (1, 8), object_poses: dict[name -> (pos (1,3), quat_wxyz (1,4))] or {}).

        arm_joint_targets covers all 8 joints AmovDrone's own arm_joint_indices
        drives (joint1-6, endleft, endright), in that order."""
        raise NotImplementedError


class ArmReplayController(ArmController):
    """Replays one recorded raids-am-simulator episode: arm joint angles,
    gripper, and all 4 object poses, via PD position targets -- same
    mechanism OmniDrones' own ``scripts/run_shelf_grasp.py`` /
    ``hover.py``'s ``arm_replay`` feature uses. Takes an explicit **frame
    index** (not a raw control-step counter) -- the recording runs at
    ``DATASET_FPS`` (30), independent of the sim's own control rate, so the
    caller must remap elapsed control steps to a frame index itself (see
    ``run_task_ShelfButtonGrasp.py``'s FLY_IN/REPLAY loop below); indexing
    1:1 by control step plays the recording back ~2x too fast at this
    sim's ~62 Hz control rate.

    Only ever plays the FIRST given episode -- no cycling between multiple
    episodes -- matching ``run_shelf_grasp.py``'s own (proven) single-episode
    design.
    """

    REQUIRED_ARM_AND_GRIPPER_JOINTS = 8  # joint1-6 + endleft + endright
    DATASET_FPS = 30.0

    def __init__(self, episode_paths):
        episode_paths = list(episode_paths)
        if not episode_paths:
            raise FileNotFoundError(
                "--arm-control=replay: no episode .npz files found/given. Pass --episode, or "
                f"populate {_EPISODES_DIR} (see docs/ASSETS.md)."
            )
        d = np.load(episode_paths[0])
        self.episode = {
            "joint_rad": torch.as_tensor(d["joint_rad"], dtype=torch.float32),
            "gripper_m": torch.as_tensor(d["gripper_m"], dtype=torch.float32),
            **{
                f"{name}_{field}": torch.as_tensor(d[f"{name}_{field}"], dtype=torch.float32)
                for name in OBJECT_NAMES
                for field in ("pos", "quat_wxyz")
            },
        }
        self.num_frames = self.episode["joint_rad"].shape[0]

    def to(self, device) -> None:
        """Move every loaded episode tensor to ``device`` -- call once the
        robot's own device is known (the data loads on CPU at construction,
        pre-Isaac-boot, but set_joint_position_targets/set_world_poses need
        it on whatever device the sim (and so the drone) actually runs on)."""
        for key, value in self.episode.items():
            self.episode[key] = value.to(device)

    def get_targets(self, frame_idx: int):
        i = max(0, min(frame_idx, self.num_frames - 1))
        ep = self.episode

        joint_rad = ep["joint_rad"][i]  # (6,)
        g = ep["gripper_m"][i]
        # endleft/endright, see airbot_play_gripper.py / AmovDrone's own
        # 8-joint detection order (joint1-6, endleft, endright).
        arm_targets = torch.cat([joint_rad, torch.stack([g, -g])]).unsqueeze(0)  # (1, 8)

        object_poses = {
            name: (ep[f"{name}_pos"][i : i + 1], ep[f"{name}_quat_wxyz"][i : i + 1])
            for name in OBJECT_NAMES
        }
        return arm_targets, object_poses


class ArmPolicyController(ArmController):
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "--arm-control=policy: no manipulation policy has been trained for this robot's "
            "action/observation layout yet. Extension point -- train one first."
        )


class ArmDeviceController(ArmController):
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "--arm-control=device: needs an airbot_play-compatible physical leader; none exists "
            "yet. Extension point -- see examples/amteleop_soarm_leader.py's SOArmLeader, which "
            "is built for the SO-ARM family, not airbot_play."
        )


def build_drone_controller(args) -> DroneController:
    if args.drone_control == "policy":
        return DronePolicyController(args.drone_policy_ckpt, ROBOT_POS, ROBOT_YAW_DEG)
    if args.drone_control == "device":
        return DroneDeviceController()
    raise ValueError(f"Unknown --drone-control {args.drone_control!r}")


def build_arm_controller(args) -> ArmController:
    if args.arm_control == "replay":
        episode_paths = [args.episode] if args.episode else DEFAULT_EPISODE_PATHS
        return ArmReplayController(episode_paths)
    if args.arm_control == "policy":
        return ArmPolicyController()
    if args.arm_control == "device":
        return ArmDeviceController()
    raise ValueError(f"Unknown --arm-control {args.arm_control!r}")


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--drone-control", choices=["policy", "device"], default="policy")
parser.add_argument("--arm-control", choices=["replay", "policy", "device"], default="replay")
parser.add_argument("--drone-policy-ckpt", type=str, default=DEFAULT_DRONE_POLICY_CKPT)
parser.add_argument(
    "--episode", type=str, default=None,
    help="Episode .npz for --arm-control=replay. Default: the first episode under "
    f"{_EPISODES_DIR}.",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--max_steps", type=int, default=0, help="0 = run until the window closes / Ctrl+C.")
# FLY_IN -> REPLAY phase thresholds, same defaults as OmniDrones's own
# scripts/run_shelf_grasp.py: hold the arm at the replay's frame 0 and let
# the hover policy fly to/settle at its target BEFORE starting the replay,
# rather than disturbing the drone with arm motion from step 0.
parser.add_argument("--pos-err-thresh", type=float, default=0.05)
parser.add_argument("--vel-thresh", type=float, default=0.1)
parser.add_argument("--settle-steps-required", type=int, default=15)
parser.add_argument("--fly-in-timeout-steps", type=int, default=1000)
args = parser.parse_args()

# Build (or fail on) both control sources before paying for an Isaac Sim boot --
# neither constructor above needs Isaac Sim, and the NotImplementedError paths
# should fail immediately, same ordering rationale as amteleop_soarm_leader.py's
# "connect the device before opening the GUI".
drone_controller = build_drone_controller(args)
arm_controller = build_arm_controller(args)

simulation_app = init_simulation_app({"headless": args.headless})

import traceback

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

from omni_drones.robots.drone.amov_drone import AmovDrone

from aetos_multiam.objects import Gearwheel, PowerDrill, ShelfButton, ToteBox
from aetos_multiam.scenes import ShelfArena
from aetos_multiam.utils import disable_collision, make_kinematic, yaw_deg_to_quat


def build_scene(make_props_kinematic: bool):
    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(2.0, 2.0, 5.0))
    prim_utils.create_prim("/World/Light/GreySphere2", "SphereLight", translation=(-2.0, -2.0, 5.0))

    # ShelfArena.spawn() (via Arena.spawn()) already creates the ground plane --
    # don't also create one here.
    ShelfArena().spawn("/World")
    if make_props_kinematic:
        # The recorded arm motion reaches right onto the shelf's own tiers to
        # grasp/place the button -- the gripper (a real, dynamic, PD-driven
        # body) can end up contacting the shelf's still-collidable geometry
        # at that point. A dynamic body contacting an immovable kinematic
        # shelf is the same force-explosion mechanism as the props (see
        # below), just against the furniture instead of a prop.
        disable_collision("/World/shelf")

    props = {
        "gearwheel": Gearwheel(name="gearwheel", translation=GEARWHEEL_POS),
        # dict key "button" matches OBJECT_NAMES / the recorded npz column names
        # (button_pos/button_quat_wxyz) -- the spawned prim name can differ.
        "button": ShelfButton(name="shelf_button", translation=BUTTON_POS),
        "power_drill": PowerDrill(name="power_drill", translation=DRILL_POS, yaw_deg=DRILL_YAW_DEG),
        "tote_box": ToteBox(name="tote_box", translation=TOTE_BOX_POS, yaw_deg=TOTE_BOX_YAW_DEG),
    }
    for obj in props.values():
        prim_path = obj.spawn("/World")
        # Flip kinematic BEFORE sim.reset(), not after: with the GPU PhysX
        # pipeline (use_gpu_pipeline=True), a live USD edit of the kinematic
        # flag after reset doesn't actually reach the already-compiled GPU
        # solver -- the body stays dynamic under the hood, and every
        # replayed set_world_poses() teleport then reads as a huge
        # instantaneous move, which PhysX resolves as a violent collision
        # (observed as constant jitter and objects getting launched away).
        # Authoring it before the first reset, like the shelf's own
        # ShelfArena.spawn(), compiles it in correctly from the start.
        if make_props_kinematic:
            make_kinematic(prim_path)
            # A teleported (kinematic) prop can't yield to contact -- if a
            # PD-driven gripper closes on it at even a slightly different
            # pose than the one it's independently being teleported to,
            # the mismatch becomes a real, undamped contact force. Replay
            # here means matching the recording's own poses, not real
            # contact dynamics, so collision comes off entirely.
            disable_collision(prim_path)
    return props


try:
    # GPU PhysX, unlike amteleop_soarm_leader.py's CPU-only demo: the shelf
    # props' collision meshes use SDF approximation, which PhysX refuses to
    # add to a CPU scene ("Rigid actors with SDFs are currently only
    # supported with GPU-accelerated scenes!").
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.016,
        rendering_dt=0.016,
        backend="torch",
        physics_prim_path="/World/PhysicsScene",
        device="cuda:0",
        sim_params={"use_gpu": True, "use_gpu_pipeline": True, "gravity": (0.0, 0.0, -9.81)},
    )
    make_props_kinematic = isinstance(arm_controller, ArmReplayController)
    props = build_scene(make_props_kinematic)

    PRIM_PATH = "/World/amov_drone_arm"
    drone = AmovDrone(name="amov_drone_arm")
    # Fly-in offset: spawn slightly off the actual hover target (same +0.3m
    # Y offset scripts/run_shelf_grasp.py uses), so the policy has to visibly
    # fly to and settle at the target before arm replay starts, rather than
    # the arm's PD correction being the very first thing that happens.
    FLY_IN_OFFSET = (0.0, 0.3, 0.0)
    spawn_pos = tuple(a + b for a, b in zip(ROBOT_POS, FLY_IN_OFFSET))
    drone.spawn(translations=[spawn_pos], orientations=[yaw_deg_to_quat(ROBOT_YAW_DEG)], prim_paths=[PRIM_PATH])

    if isinstance(arm_controller, ArmReplayController):
        arm_controller.to(drone.device)

    sim.reset()
    drone.initialize(prim_paths_expr=PRIM_PATH)
    for obj in props.values():
        obj.initialize(f"/World/{obj.name}")

    if isinstance(arm_controller, ArmReplayController) and drone.num_arm_joints != ArmReplayController.REQUIRED_ARM_AND_GRIPPER_JOINTS:
        raise RuntimeError(
            f"--arm-control=replay needs {ArmReplayController.REQUIRED_ARM_AND_GRIPPER_JOINTS} "
            f"arm+gripper joints (the recorded data's own shape); this robot has "
            f"{drone.num_arm_joints}."
        )

    # Match the arm's PD gains to whatever the policy was actually trained
    # against (fix 5 in the module docstring) -- rather than whatever
    # default drive gains amov_drone_arm_1.usd happens to author. Both
    # training and scripts/run_shelf_grasp.py set this explicitly in code
    # regardless of the USD's own authored defaults.
    if isinstance(arm_controller, ArmReplayController):
        n = drone.num_arm_joints
        stiffness = torch.full((drone._view.count, n), ARM_GAIN_STIFFNESS, device=drone.device)
        damping = torch.full((drone._view.count, n), ARM_GAIN_DAMPING, device=drone.device)
        drone._view.set_gains(stiffness, damping, joint_indices=drone.arm_joint_indices)

    # Teleport the arm to the replay's own starting frame BEFORE the first
    # physics step -- position, velocity (zeroed), and PD target all at
    # once, via AmovDrone's own reset_arm_joints (fix 3 in the module
    # docstring). Skipping this leaves the arm at its USD rest pose while
    # the PD target jumps straight to the recorded frame 0 on step 0.
    initial_arm_targets, _ = arm_controller.get_targets(0)
    drone.reset_arm_joints(torch.tensor([0], device=drone.device), initial_arm_targets)

    # Hover action normalization (hover.py:589-624, applied unconditionally
    # in _pre_sim_step for any task without a rate/attitude controller
    # transform -- which includes HoverAmovDynaArmReplay): the policy was
    # trained with its raw [-1,1] action remapped so action=0 maps to the
    # exact hover thrust command, not to the RotorGroup's own raw cmd=0.
    # Feeding the policy's raw action straight into apply_action (skipping
    # this) is a silent, systematic thrust bias -- consistent with the
    # steady-state position offset and reduced disturbance-robustness
    # observed before this was added.
    hover_cmd_offset = hover_cmd_scale = None
    if isinstance(drone_controller, DronePolicyController):
        with torch.no_grad():
            _g = drone.gravity.mean()
            _kf = drone.KF.sum(-1).mean().clamp_min(1e-9)
            _hover_frac = (_g / _kf).clamp(0.0, 1.0)
            hover_cmd_offset = float((2.0 * _hover_frac - 1.0).item())
            hover_cmd_scale = float(1.0 - abs(hover_cmd_offset))
        print(
            f"[run_task_ShelfButtonGrasp] hover action normalization: "
            f"hover_frac={_hover_frac.item():.4f} offset={hover_cmd_offset:.4f} "
            f"scale={hover_cmd_scale:.4f}",
            flush=True,
        )

    sim.step(render=False)

    vp = get_active_viewport()
    cam_state = ViewportCameraState("/OmniverseKit_Persp", vp)
    cam_state.set_position_world(Gf.Vec3d(2.5, -2.5, 3.5), True)
    cam_state.set_target_world(Gf.Vec3d(0.0, 0.0, 1.3), True)
except Exception:
    traceback.print_exc()
    simulation_app.close()
    raise

print(
    f"[run_task_ShelfButtonGrasp] robot_usd={AMOV_DRONE_ARM_USD_PATH} "
    f"drone-control={args.drone_control} arm-control={args.arm_control}",
    flush=True,
)

# FLY_IN -> REPLAY phase machine, ported from scripts/run_shelf_grasp.py:
# hold the arm at the replay's frame 0 while the policy flies in from its
# offset spawn and settles (position error + speed both under threshold
# for settle_steps_required consecutive steps, or fly_in_timeout_steps
# elapses) -- only THEN does the replay frame index start advancing.
replay_enabled = isinstance(arm_controller, ArmReplayController)
phase = "FLY_IN" if replay_enabled else "REPLAY"
settle_counter = 0
replay_start_step = None
frame_idx = 0
control_hz = 1.0 / sim.get_physics_dt()
target_pos_tensor = torch.tensor([list(ROBOT_POS)], device=drone.device)  # (1, 3)

step = 0
try:
    while simulation_app.is_running():
        drone_action = drone_controller.get_action(drone, step)
        if hover_cmd_scale is not None:
            drone_action = (drone_action * hover_cmd_scale + hover_cmd_offset).clamp(-1.0, 1.0)

        if replay_enabled and phase == "REPLAY":
            elapsed = step - replay_start_step
            frame_f = elapsed * (ArmReplayController.DATASET_FPS / control_hz)
            frame_idx = min(int(frame_f), arm_controller.num_frames - 1)
        arm_targets, object_poses = arm_controller.get_targets(frame_idx)

        drone.apply_action(drone_action)
        # hold_arm_joints(zero_velocity=True) sets both the PD target AND
        # zeroes joint velocity every step (not just once at init) -- same
        # as hover.py's own _pre_sim_step sequence (fix 4b in the module
        # docstring). Damps any PD overshoot before it can build up and
        # destabilize the drone.
        drone.hold_arm_joints(arm_targets, zero_velocity=True)
        for name, (pos, quat) in object_poses.items():
            props[name]._view.set_world_poses(positions=pos, orientations=quat)

        sim.step(render=True)
        step += 1

        drone_state = drone.get_state()
        pos_err = torch.norm(target_pos_tensor - drone_state[..., :3]).item()
        speed = torch.norm(drone_state[..., 7:10]).item()
        up_z = float(drone_state[..., 18].reshape(-1)[0])

        if up_z < 0.5:
            print(
                f"[run_task_ShelfButtonGrasp] drone appears to have flipped "
                f"(up_z={up_z:.3f}) at step={step} -- stopping.",
                flush=True,
            )
            break

        if replay_enabled and phase == "FLY_IN":
            if pos_err < args.pos_err_thresh and speed < args.vel_thresh:
                settle_counter += 1
            else:
                settle_counter = 0
            if settle_counter >= args.settle_steps_required or step >= args.fly_in_timeout_steps:
                phase = "REPLAY"
                replay_start_step = step
                print(
                    f"[run_task_ShelfButtonGrasp] FLY_IN -> REPLAY at step={step} "
                    f"pos_err={pos_err:.3f} (timeout={step >= args.fly_in_timeout_steps})",
                    flush=True,
                )

        if step % 50 == 0:
            print(
                f"[run_task_ShelfButtonGrasp] step={step} phase={phase} "
                f"drone_pos={drone_state[..., :3].reshape(-1).tolist()} "
                f"pos_err={pos_err:.3f} speed={speed:.3f} up_z={up_z:.3f}",
                flush=True,
            )
        if args.max_steps and step >= args.max_steps:
            break
except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
