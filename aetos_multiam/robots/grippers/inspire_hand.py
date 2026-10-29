"""Inspire Hand -- two real variants, extracted from larger public USD scenes.

No official standalone Inspire Hand USD was found anywhere (not from Inspire
Robotics themselves, not via Isaac Sim's own "OpenUSD tuning tutorials"
sample content, which uses an Inspire Hand RH56DFX as its teaching example
but ships only a URDF bundled with a newer Isaac Sim installer, not a
downloadable USD -- see docs/ARCHITECTURE.md). What exists instead: two
*other* public USD scenes that happen to embed a real Inspire Hand as part
of a bigger robot, found in github.com/konu-droid/robots_usd (downloaded
with the user's explicit go-ahead, then the hand's own body/joint/mesh
subtree extracted out into a standalone file -- see
``aetos_multiam/assets/usd/inspire_hand/`` for the extraction script and
its own docstring for the two non-obvious USD pitfalls hit along the way:
an internal mesh-dedup scheme using same-layer `over`-specifier prims that
neither `Stage.Flatten()` nor `Usd.PrimRange`'s default predicate reach
into, and `Sdf.CopySpec()` never remapping relationship target paths).

**Variant 1: `inspire_hand_g1dfq_{left,right}`** -- extracted from
`unitree/g1/g1_hands/g1_hands.usd` (the bimanual hands of a full Unitree G1
humanoid). The source file's own internal name is literally
`g1_29dof_rev_1_0_with_inspire_hand_DFQ` -- "DFQ" per the source, not a
number this project independently verified against an Inspire Robotics
spec sheet. 12 joints/hand, all independently `PhysicsRevoluteJoint`s (no
coupled/fixed segments) -- index/middle/ring/pinky each have a proximal +
intermediate joint, thumb has proximal_yaw + proximal_pitch + intermediate
+ distal. Verified live: 12-DOF Articulation, real textured geometry
renders (not placeholder boxes).

**Variant 2: `inspire_hand_kuka_right`** -- extracted from
`kuka/kuka_w_hand.usd` (a Kuka-arm + single right-hand demo scene; this
repo's README and an NVIDIA doc both mention "Kuka and Inspire Hand" as a
known paired demo). Also 12 `PhysicsRevoluteJoint`s, but a different,
underactuated-looking topology: 5 proximal joints directly off the palm
(`index_bend`/`mid_joint1`/`pinky_joint1`/`ring_joint1`/`thumb_bend`) + 7
more further down each finger chain, with each fingertip's LAST segment
attached via a `PhysicsFixedJoint` (not independently actuated) -- a
different Inspire Hand product variant from the G1 one, not re-derived or
assumed identical.

Registered the same way `so_arm100_gripper` is: no ``ManipulatorSpec``
wrapper, since these are standalone end-effectors meant to mount directly
onto whatever arm's ``eef_body_name`` via ``mount_child`` -- a pure
``GripperSpec`` with a real ``usd_path``/``root_body_name`` is the whole
story.

Open/closed values: ``GripperSpec.format_action`` only supports one scalar
in [-1, 1] mapped uniformly across all listed joints (same simplification
used by every other gripper in this project) -- not real per-finger
dexterity. Values here are each joint's own authored USD lower/upper limit
(open=lower, closed=upper, converted degrees->radians since Isaac Sim's
runtime joint-position API uses radians regardless of how USD authors
limits), i.e. "fully open hand" to "fully closed fist" within each joint's
own real range -- not fine-tuned against an actual grasp test.
"""

import os.path as osp

from aetos_multiam.assets import USD_PATH as _ASSETS_USD_DIR
from .gripper_base import GripperSpec, register_gripper

_HAND_USD_DIR = osp.join(_ASSETS_USD_DIR, "inspire_hand")

G1DFQ_LEFT_USD_PATH = osp.join(_HAND_USD_DIR, "inspire_hand_g1dfq_left.usd")
G1DFQ_RIGHT_USD_PATH = osp.join(_HAND_USD_DIR, "inspire_hand_g1dfq_right.usd")
KUKA_RIGHT_USD_PATH = osp.join(_HAND_USD_DIR, "inspire_hand_kuka_right.usd")

INSPIRE_HAND_G1DFQ_LEFT = register_gripper(
    GripperSpec(
        name="inspire_hand_g1dfq_left",
        joint_names=(
            "L_index_proximal_joint", "L_index_intermediate_joint",
            "L_middle_proximal_joint", "L_middle_intermediate_joint",
            "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
            "L_ring_proximal_joint", "L_ring_intermediate_joint",
            "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint", "L_thumb_distal_joint",
        ),
        open_joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.0),
        closed_joint_positions=(1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.3, 0.6, 0.8, 1.2),
        usd_path=G1DFQ_LEFT_USD_PATH,
        root_body_name="hand_base",
        note=(
            "Extracted from konu-droid/robots_usd's unitree/g1/g1_hands/g1_hands.usd "
            "(left hand). 'DFQ' per that source file's own internal name. "
            "Mount translation/orientation onto an arm's eef_body_name not yet derived."
        ),
    )
)

INSPIRE_HAND_G1DFQ_RIGHT = register_gripper(
    GripperSpec(
        name="inspire_hand_g1dfq_right",
        joint_names=(
            "R_index_proximal_joint", "R_index_intermediate_joint",
            "R_middle_proximal_joint", "R_middle_intermediate_joint",
            "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
            "R_ring_proximal_joint", "R_ring_intermediate_joint",
            "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint", "R_thumb_distal_joint",
        ),
        open_joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.0),
        closed_joint_positions=(1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.3, 0.6, 0.8, 1.2),
        usd_path=G1DFQ_RIGHT_USD_PATH,
        root_body_name="hand_base",
        note=(
            "Extracted from konu-droid/robots_usd's unitree/g1/g1_hands/g1_hands.usd "
            "(right hand). 'DFQ' per that source file's own internal name. "
            "Mount translation/orientation onto an arm's eef_body_name not yet derived."
        ),
    )
)

INSPIRE_HAND_KUKA_RIGHT = register_gripper(
    GripperSpec(
        name="inspire_hand_kuka_right",
        joint_names=(
            "robot1_gripper_right_hand_index_bend_joint", "robot1_gripper_right_hand_index_joint1",
            "robot1_gripper_right_hand_index_joint2",
            "robot1_gripper_right_hand_mid_joint1", "robot1_gripper_right_hand_mid_joint2",
            "robot1_gripper_right_hand_pinky_joint1", "robot1_gripper_right_hand_pinky_joint2",
            "robot1_gripper_right_hand_ring_joint1", "robot1_gripper_right_hand_ring_joint2",
            "robot1_gripper_right_hand_thumb_bend_joint",
            "robot1_gripper_right_hand_thumb_rota_joint1", "robot1_gripper_right_hand_thumb_rota_joint2",
        ),
        open_joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.05, -0.175),
        closed_joint_positions=(0.175, 1.92, 1.92, 1.92, 1.94, 1.92, 1.92, 1.92, 1.92, 1.83, 1.57, 1.83),
        usd_path=KUKA_RIGHT_USD_PATH,
        root_body_name="hand_base",
        note=(
            "Extracted from konu-droid/robots_usd's kuka/kuka_w_hand.usd (right "
            "hand only -- that scene has no left hand). A different, apparently "
            "more underactuated Inspire Hand variant from the G1 one (each "
            "fingertip's last segment is a PhysicsFixedJoint, not independently "
            "actuated -- only 12 of ~17 joints in the source are revolute, those "
            "12 are what's listed here). Mount translation/orientation not yet derived."
        ),
    )
)
