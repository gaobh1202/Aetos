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

**Variant 3: `inspire_hand_right`** -- a later, *complete* official Isaac Sim
tutorial download (``inspire_hand.usda`` + ``payloads/`` + ``Materials/``,
unlike an earlier single-file copy that was missing those and wouldn't
resolve). This is the one actually mounted onto ``airbot_dexterous`` (see
``manipulators/airbot_play.py``) -- real asset, real mount derivation, not a
placeholder. `inspire_hand_g1dfq_right` (the G1-extracted right hand) was
retired once this arrived -- redundant with this cleaner asset -- its file
and registration removed; `inspire_hand_g1dfq_left` is kept since no clean
replacement for a left hand exists yet.

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
KUKA_RIGHT_USD_PATH = osp.join(_HAND_USD_DIR, "inspire_hand_kuka_right.usd")
RIGHT_USD_PATH = osp.join(_HAND_USD_DIR, "inspire_hand_right", "inspire_hand.usda")

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

# inspire_hand_right -- a *complete* official Isaac Sim tutorial download
# (inspire_hand.usda + payloads/ + Materials/, not a single-file export missing
# those -- vendored as-is at assets/usd/inspire_hand/inspire_hand_right/).
# root_body_name="r_base_link"; 12 independently-actuated PhysicsRevoluteJoints
# (thumb_1..4, {index,middle,ring,little}_{1,2}), every fingertip's last
# segment + the palm shell welded on via PhysicsFixedJoint (not independently
# actuated) -- read directly off the live USD (UsdPhysics.RevoluteJoint's
# lower/upper limits are authored in DEGREES; GripperSpec stores radians for
# format_action, degrees for the two mount-joint limits below since those feed
# UsdPhysics.RevoluteJoint.CreateLowerLimitAttr directly).
#
# mount_translation/mount_orientation went through 4 iterations mounting
# this hand onto airbot_dexterous's link5 (see docs/ARCHITECTURE.md for the
# full derivation of each):
#   1. (0,0,0) / (0.707107, 0, -0.707107, 0) -- Fixed mount onto a
#      custom_end_link frame. Superseded once the whole link6 (not just its
#      fingers) was removed and the hand took over link5's own wrist joint6.
#   2. (0, 0.23645, 0) / (0.5, 0.5, 0.5, -0.5) -- Revolute mount reusing
#      joint6's own localPos0/localRot0 directly, plus a spin about the
#      hand's own LOCAL Z (misreading "绕z轴旋转180度" as the hinge axis
#      instead of world Z) -- wrong, fingers ended up pointing at link5.
#   3. (0.0, 0.23645, 0.0) / (0.5, -0.5, 0.5, 0.5) -- same translation,
#      rotation recomposed about WORLD Z instead (see mount_child's docstring
#      for the Gf.Matrix3d-vs-Matrix4d.TransformDir sign trap found while
#      debugging this). Fixed the finger direction but left a ~0.15m gap
#      (joint6's offset assumed a part shaped like the old link6, whose own
#      mesh overlaps backward into link5 -- r_base_link's mesh doesn't) and
#      the palm faced up.
#   4. FINAL, current values below -- translation reduced so r_base_link's
#      origin sits exactly at link5's own mesh edge (measured live via
#      UsdGeom.BBoxCache: link5 ended at world x=0.1304, r_base_link's
#      origin needed to land there, not at joint6's old x=0.282), plus an
#      additional 180 degrees about WORLD X (the arm's reach axis) composed
#      on top of attempt 3's rotation to flip the palm from up to down.
#      Verified live: local Z (fingers) still -> world +X unaffected, local
#      X (palm normal, established by comparing each fingertip's pad vs tip
#      offset in the standalone asset) now -> world -Z (down).
INSPIRE_HAND_RIGHT = register_gripper(
    GripperSpec(
        name="inspire_hand_right",
        joint_names=(
            "right_thumb_1_joint", "right_thumb_2_joint",
            "right_thumb_3_joint", "right_thumb_4_joint",
            "right_index_1_joint", "right_index_2_joint",
            "right_middle_1_joint", "right_middle_2_joint",
            "right_ring_1_joint", "right_ring_2_joint",
            "right_little_1_joint", "right_little_2_joint",
        ),
        open_joint_positions=(0.0,) * 12,
        closed_joint_positions=(1.309, 0.5236, 0.3578, 0.2775, 1.344, 1.527, 1.344, 1.527, 1.344, 1.527, 1.344, 1.527),
        usd_path=RIGHT_USD_PATH,
        root_body_name="r_base_link",
        mount_translation=(0.0, 0.0848, 0.0),
        mount_orientation=(0.5, -0.5, -0.5, -0.5),
        mount_joint_type="Revolute",
        mount_joint_axis="Z",
        mount_joint_lower_limit=-172.86135864257812,
        mount_joint_upper_limit=172.86135864257812,
        note=(
            "Complete official Isaac Sim tutorial Inspire Hand download "
            "(inspire_hand.usda + payloads/ + Materials/). Mounts onto "
            "airbot_dexterous's link5, taking over the same wrist DOF "
            "airbot_gripper's link6/joint6 used to occupy -- see the mount "
            "derivation history above and docs/ARCHITECTURE.md."
        ),
    )
)
