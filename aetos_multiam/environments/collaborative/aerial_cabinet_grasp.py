"""AerialCabinetGrasp -- two AMs pull a cabinet's doors open, a third takes a button out.

Scene (``WarehouseArena``, concrete floor, no gantry):

* a two-door **cabinet** (SimReady ``D00241``, ``objects/cabinet.py``) standing
  between the two painted spawn zones, its front facing -y, both doors shut;
* a **button** (``ShelfButton``) on one of its shelves, behind those doors;
* two ``fahexa750_gripper``, one per zone, and one
  ``hummingbird340_2dof_gripper`` in the lane in front of the cabinet.

Intended flow: each hexacopter flies to the handle of the door on its own
side, closes on it, and swings the door open by following the arc the handle
sweeps about its hinge; with both doors held open the Hummingbird flies into
the opening, lowers its gripper onto the button and carries it out.

The split of roles is the point. A door handle wants reach and pulling force,
which is the 750 mm hexacopter with its top-mounted 4-DOF arm (and, being
fully actuated, it can pull sideways without tilting). The compartment the
button sits in is too small for that airframe to go anywhere near, and is
what the 340 mm quadrotor with a short arm hanging under it is for. Agent
order matters: **agent 0 takes the -x door, agent 1 the +x door, agent 2 the
button** -- the same -x-first order the arena lists its zones in and
``Cabinet`` numbers its doors in.

``task.cabinet_scale`` defaults to **2**, not 1. At the asset's own size the
cabinet is 0.94 m wide with compartments 0.37 m high, which no drone here can
get an arm into. Doubled, it is a 1.9 x 2.6 m wardrobe: compartments 1.41 m
wide and 0.74 m high, handles a 46 mm bar at 1.38 m -- inside the hexa
gripper's 90 mm stroke, and at a height its arm reaches from a comfortable
hover.

The waypoints the flow needs -- ``handle_position``,
``handle_hover_position``, ``button_position``, ``button_hover_position`` --
are methods here rather than numbers buried in a reward, so whatever ends up
driving the robots reads them from one place.

Not implemented yet: the grasps themselves and a controller for the arms.
Reward is ``CollaborativeAerialEnv``'s placeholder (each agent holds position
near its own goal) with the goals set to where each robot has to *end up* --
the hexacopters beside their open doors, the Hummingbird over the button --
plus, for the two door agents, how far their own door has actually opened.
Success additionally requires both doors past ``task.door_open_threshold``,
so hovering at the goals without having moved the doors does not count.
"""

from typing import List, Optional, Sequence

import torch

from aetos_multiam.objects import Cabinet, ShelfButton
from aetos_multiam.scenes import WarehouseArena
from aetos_multiam.utils import yaw_deg_to_quat

from .collaborative_base import CollaborativeAerialEnv

SPAWN_HEIGHT_M = 1.2

#: The cabinet stands on the bay's centreline, pushed back along +y so that
#: its doors, swung fully open, still stop short of the line between the two
#: spawn zones.
CABINET_XY_M = (0.0, 1.0)

#: Where the Hummingbird starts: out in the lane the cabinet's front faces.
RETRIEVER_SPAWN_XY_M = (0.0, -3.5)

#: Agents 0 and 1 open doors 0 and 1; agent 2 fetches the button.
NUM_DOOR_AGENTS = 2
RETRIEVER_AGENT = 2

#: Where a hexacopter's body sits relative to the handle it is holding: this
#: far back along the handle's normal and this far below it. Its arm is
#: mounted on top and puts the gap between its fingers up to about 1.1 m
#: forward (``data/arm_kinematics/fahexa750_gripper.npz``); 0.75 m is well
#: inside that and leaves a quarter of a metre between the door and the
#: nearest propeller tip, which 0.55 m did not.
HANDLE_STANDOFF_M = 0.75
HANDLE_BELOW_M = 0.25

#: How far above the shelf the Hummingbird's body hovers to hold the button.
#: With its arm hanging straight down the point its three fingers close on is
#: 0.385 m under the airframe (``data/arm_kinematics/
#: hummingbird340_2dof_gripper.npz`` plus the finger meshes), and what they
#: close on is the button's cap, centred 66 mm above the shelf.
BUTTON_HOVER_M = 0.451

#: Robot links the fixtures stop colliding with when ``task.filter_grasp_contacts``
#: is on: the hexacopters' gripper against the doors, the Hummingbird's against
#: the button.
HEXA_GRIPPER_LINKS = ("arm_link4_roll_link", "ee_link",
                      "manipulation_lfinger_link", "manipulation_rfinger_link")
RETRIEVER_GRIPPER_LINKS = ("Housing", "finger_1", "finger_2", "finger_3")


class AerialCabinetGrasp(CollaborativeAerialEnv):
    def __init__(self, cfg, headless):
        self.cabinet_scale = float(cfg.task.get("cabinet_scale", 2.0))
        self.button_compartment = int(cfg.task.get("button_compartment", 1))
        self.button_depth_m = float(cfg.task.get("button_depth_m", 0.15))
        self.door_open_threshold = float(cfg.task.get("door_open_threshold", 0.8))
        self.door_reward_weight = float(cfg.task.get("door_reward_weight", 1.0))
        if len(cfg.task.aerial_manipulators) != NUM_DOOR_AGENTS + 1:
            raise ValueError(
                "AerialCabinetGrasp needs exactly three aerial manipulators -- one per "
                f"door, then one for the button -- got {len(cfg.task.aerial_manipulators)}."
            )
        # Built here rather than in _load_objects: the robots are spawned
        # first, and where they face depends on where the cabinet is.
        self.cabinet = Cabinet(
            name="cabinet",
            translation=(CABINET_XY_M[0], CABINET_XY_M[1], 0.0),
            scale=self.cabinet_scale,
        )
        super().__init__(cfg, headless)

    # -- scene ---------------------------------------------------------------

    def _build_arena(self) -> WarehouseArena:
        return WarehouseArena(floor_color=WarehouseArena.CONCRETE, gantry=False)

    def _default_spawn_translation(self, index: int) -> Sequence[float]:
        if index == RETRIEVER_AGENT:
            return (*RETRIEVER_SPAWN_XY_M, SPAWN_HEIGHT_M)
        centers = self.arena.spawn_zone_centers
        cx, cy = centers[index % len(centers)]
        return (cx, cy, SPAWN_HEIGHT_M)

    def _default_spawn_orientation(self, index: int) -> Optional[Sequence[float]]:
        """Everyone starts facing the cabinet: the hexacopters' arms reach
        along their own +x, so this is the heading they pull from."""
        if index == RETRIEVER_AGENT:
            return yaw_deg_to_quat(90.0)
        return yaw_deg_to_quat(0.0 if index == 0 else 180.0)

    # -- waypoints the flow needs -------------------------------------------

    def handle_position(self, index: int, open_fraction: float = 0.0) -> Sequence[float]:
        """Centre of door ``index``'s handle bar. ``open_fraction`` 0 is the
        door shut, 1 fully open, in between the arc a pull has to follow."""
        return self.cabinet.handle_position(index, open_fraction)

    def handle_hover_position(self, index: int, open_fraction: float = 0.0) -> Sequence[float]:
        """Where a hexacopter's body has to be to hold that handle."""
        x, y, z = self.cabinet.handle_approach_position(index, HANDLE_STANDOFF_M, open_fraction)
        return (x, y, z - HANDLE_BELOW_M)

    @property
    def button_position(self) -> Sequence[float]:
        """Where the button stands: the middle of its shelf, a little way in
        from the front edge."""
        return self.cabinet.shelf_position(
            self.button_compartment, x_fraction=0.5, depth_m=self.button_depth_m)

    @property
    def button_hover_position(self) -> Sequence[float]:
        x, y, z = self.button_position
        return (x, y, z + BUTTON_HOVER_M)

    # -- prims ---------------------------------------------------------------

    def _load_objects(self) -> List[str]:
        ns = self.template_env_ns
        self.cabinet.spawn(ns)

        bx, by, bz = self.button_position
        self.button = ShelfButton(name="shelf_button", translation=(bx, by, bz + 0.002))
        self.button.spawn(ns)
        if self.cfg.task.get("filter_grasp_contacts", False):
            self._filter_grasp_contacts()

        # Everything here is inside the template env and gets cloned.
        return []

    def _filter_grasp_contacts(self) -> None:
        """Gripper <-> fixture collisions off (authored before the sim starts).

        For a scripted run on the frozen-arm plant, where the door is turned
        and the button carried KINEMATICALLY from the gripper's pose: a
        gripper link there weighs a few grams, and contact between it and a
        fixture that is being teleported does not hold the fixture -- it bends
        the arm and drags the drone off its setpoint (measured on
        ``AerialTransport``, which filters its box for the same reason). The
        grasp is then judged geometrically, see
        ``scripts/run_cabinet_grasp_demo.py``. Off by default: a policy that
        is meant to feel the handle needs these contacts."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd, UsdPhysics

        from aetos_multiam.objects.cabinet import ARTICULATION_NAME

        stage = get_current_stage()
        ns = self.template_env_ns

        def bodies(root_path, names):
            found = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root_path))
                     if p.GetName() in names and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(found) != len(names):
                raise RuntimeError(
                    f"AerialCabinetGrasp: expected bodies {names} under {root_path}, found {found}")
            return found

        # Every door against BOTH hexacopters' grippers, not just its own: with
        # the doors shut the two handles are 10 cm apart, so a gripper open on
        # one of them is within a centimetre of the other.
        hexa_grippers = [path for agent in range(NUM_DOOR_AGENTS)
                         for path in bodies(f"{ns}/am_{agent}_0", HEXA_GRIPPER_LINKS)]
        for door in range(NUM_DOOR_AGENTS):
            door_prim = stage.GetPrimAtPath(
                f"{ns}/{self.cabinet.name}/{ARTICULATION_NAME}/{self.cabinet.door_body_names[door]}")
            rel = UsdPhysics.FilteredPairsAPI.Apply(door_prim).CreateFilteredPairsRel()
            for path in hexa_grippers:
                rel.AddTarget(path)

        gripper = bodies(f"{ns}/am_{RETRIEVER_AGENT}_0", RETRIEVER_GRIPPER_LINKS)
        for body_name in self.button.root_body_names:
            rel = UsdPhysics.FilteredPairsAPI.Apply(
                stage.GetPrimAtPath(f"{ns}/{self.button.name}/{body_name}")).CreateFilteredPairsRel()
            for path in gripper:
                rel.AddTarget(path)

    def _setup_object_references(self) -> None:
        self.cabinet.initialize(f"{self.env_ns}/env_.*/cabinet")
        self.button.initialize(f"{self.env_ns}/env_.*/shelf_button")

    # -- task ----------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.cabinet.reset_doors(env_ids)

    def _goal_position(self, index: int) -> Sequence[float]:
        """Where each robot has to end up: the hexacopters beside their own
        door held fully open, the Hummingbird over the button."""
        if index == RETRIEVER_AGENT:
            return self.button_hover_position
        return self.handle_hover_position(index, open_fraction=1.0)

    def reward(self) -> dict:
        rewards = super().reward()
        door_open = self.cabinet.get_door_open_fraction()          # (num_envs, 2)
        for door in range(NUM_DOOR_AGENTS):
            key = self._agent_keys[door]
            rewards[key] = rewards[key] + self.door_reward_weight * door_open[:, door:door + 1]
        return rewards

    def _check_success(self) -> torch.Tensor:
        doors_open = (self.cabinet.get_door_open_fraction() >= self.door_open_threshold).all(
            dim=-1, keepdim=True)
        return super()._check_success() & doors_open
