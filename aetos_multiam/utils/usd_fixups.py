"""USD authoring fixups needed when referencing pre-baked rigid-body props
into a scene -- ported from raids-am-simulator's shelf_grasp scene builder
(``raids_am_Scene/build_isaacsim_scene.py``), which this project's
``scenes.ShelfArena``/``objects.{Gearwheel,ShelfButton,PowerDrill,ToteBox}``
were migrated from.
"""

import logging
import math

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as physx_script_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

get_current_stage = prim_utils.get_current_stage


def yaw_deg_to_quat(yaw_deg: float):
    """Z-axis-only rotation as a (w, x, y, z) quaternion -- every prop/robot
    pose migrated from the shelf_grasp scene is only ever yawed, never
    pitched/rolled."""
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def normalize_rigid_root_xform(prim_path: str, scale=(1.0, 1.0, 1.0)) -> None:
    """Re-author a referenced rigid body's xform ops as an equivalent
    translate/rotateXYZ/scale triplet.

    These source assets each bake a single non-standard xformOp on their
    rigid root (e.g. one combined 'transform' matrix, or translate/
    rotateZYX/scale) -- PhysX can't append the op it needs for simulated
    pose updates on top of that, and the body silently never moves
    ("Unable to add xform op" warnings in a live run). ``scale`` lets a
    caller re-apply a scale this same prim needs (e.g. the shelf's 1.5x
    local-X widening) since normalizing would otherwise drop it.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"normalize_rigid_root_xform: prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    local_xform = xf.GetLocalTransformation()
    translate = local_xform.ExtractTranslation()
    euler_xyz = local_xform.ExtractRotation().Decompose(
        Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()
    )
    for op in xf.GetOrderedXformOps():
        prim.RemoveProperty(op.GetOpName())
    xf.ClearXformOpOrder()
    common = UsdGeom.XformCommonAPI(prim)
    common.SetTranslate(translate)
    common.SetRotate(Gf.Vec3f(euler_xyz[0], euler_xyz[1], euler_xyz[2]))
    common.SetScale(Gf.Vec3f(*scale))


def make_kinematic(root_path: str) -> int:
    """Flip every RigidBodyAPI prim under ``root_path`` to kinematic, so the
    subtree behaves as static furniture (keeps its authored collision, just
    never falls/reacts to gravity or bumps). Returns how many prims were
    flipped, so a caller can assert it actually found bodies to flip.

    Must be called BEFORE the stage's first sim.reset() under the GPU PhysX
    pipeline (use_gpu_pipeline=True): a live USD edit of the kinematic flag
    after reset doesn't reach the already-compiled GPU solver, so the body
    stays effectively dynamic and any subsequent pose teleport reads as a
    violent collision (constant jitter, body launched away).

    Also deactivates every internal PhysicsJoint prim found in the same
    subtree -- once both ends of a joint are about to be kinematic, PhysX
    refuses to create the joint at all ("cannot create a joint between
    static bodies"), which otherwise aborts the whole stage parse.
    """
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"make_kinematic: prim not found: {root_path}")
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        count += 1
    return count


def deactivate_prim(prim_path: str) -> None:
    """Deactivate a stray prim (e.g. power_drill's own world-anchoring
    FixedJoint) baked into a source asset."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"deactivate_prim: prim not found: {prim_path}")
    prim.SetActive(False)


def disable_collision(root_path: str) -> int:
    """Disable collision on every CollisionAPI prim under ``root_path``.

    For a prop whose pose is being teleported every frame to match a
    recording (``make_kinematic`` + per-frame ``set_world_poses``), the
    teleported pose is already correct -- but if the prop still collides,
    a dynamic body that only approximately reaches the same place each
    frame (a PD-driven gripper closing on it, say) generates a real
    contact force against an object that physically cannot yield (it's
    kinematic). This was observed directly: a recorded episode's gripper
    closing on an object at a specific frame coincided exactly with the
    drone flipping -- the collision reaction torque at the grasp moment,
    not the arm's own free motion, was the actual cause. Disabling
    collision makes a replayed prop purely visual (matches the recording's
    own poses, never pushes back), which is what "replay" means here --
    real contact dynamics would need a real grasping policy instead
    (ArmPolicyController, not yet implemented).
    """
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"disable_collision: prim not found: {root_path}")
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        count += 1
    return count


def pin_body_to_world(body_prim_path: str) -> None:
    """Weld a body to the world with a real PhysX Fixed joint (``body0=None``
    means "the world"), done BEFORE ``sim.reset()`` so it's part of the
    articulation's structure from the start -- ported from
    ``examples/amteleop_soarm_leader.py``'s ``pin_base_link_to_world``.

    Re-teleporting a drone's pose every step doesn't reliably hold it still:
    a merged drone+arm(+gripper) articulation's actual PhysX root can be a
    body deep in the arm chain rather than the drone's own ``base_link``
    (confirmed empirically there), so teleporting "the root" pins the wrong
    body and leaves ``base_link`` free to swing from the arm's own joint
    motion. A world-anchored Fixed joint directly on the named body
    sidesteps the root-identity question entirely -- whatever the
    articulation's root turns out to be, this body still can't move.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(body_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"pin_body_to_world: prim not found: {body_prim_path}")
    physx_script_utils.createJoint(stage, "Fixed", None, prim)


def weld_joint(joint_prim_path: str) -> None:
    """Retype a movable joint (Revolute/Prismatic/...) to a Fixed joint, at
    whatever relative pose its ``localPos0``/``localRot0``/``localPos1``/
    ``localRot1`` frames already encode -- i.e. rigidly lock two bodies
    together at their CURRENT authored relative pose, no rotation math
    needed since those frame attributes are exactly what a joint's rest
    pose already is.

    Use this instead of a live ``UsdPhysics.DriveAPI`` (``add_joint_drive``)
    when a movable joint only needs to ever sit in one fixed display pose:
    a drive's target only lives in the physics solver's own state and isn't
    guaranteed to reach the renderer every frame (confirmed empirically on
    more than one SimReady prop -- the joint's tensor-API-reported angle
    was correct throughout, but the body's rendered/USD-level transform
    either froze on the very first frame or never moved at all). A Fixed
    joint has no runtime DOF to sync in the first place, so this sidesteps
    that whole class of bug rather than working around it.

    Must be called before the first ``sim.reset()``, same timing as every
    other structural USD edit in this module.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(joint_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"weld_joint: prim not found: {joint_prim_path}")
    if not prim.IsA(UsdPhysics.Joint):
        raise RuntimeError(f"weld_joint: prim is not a UsdPhysics.Joint: {joint_prim_path}")
    # Drop the old movable-joint-specific properties (RevoluteJoint's own
    # axis/lowerLimit/upperLimit, any JointStateAPI/DriveAPI) before
    # retyping -- NOT the base Joint attributes (localPos0/localRot0/...,
    # which define the weld's rest pose and must survive). Confirmed
    # empirically that leaving the old type's own attributes behind on a
    # prim now typed PhysicsFixedJoint corrupts the stage's whole PhysX
    # parse (every rigid body tensor-API lookup in the scene starts
    # failing with "did not match any rigid bodies", not just this joint).
    for prop in list(prim.GetProperties()):
        name = prop.GetName()
        if name.startswith(("physics:axis", "physics:lowerLimit", "physics:upperLimit")):
            prim.RemoveProperty(name)
        elif prop.GetNamespace() in ("state", "drive"):
            prim.RemoveProperty(name)
    prim.SetTypeName("PhysicsFixedJoint")


def weld_joint_at_angle(joint_prim_path: str, angle_deg: float) -> None:
    """Like ``weld_joint``, but locks the joint at ``angle_deg`` (its
    ``RevoluteJoint`` ``position`` convention) instead of whatever pose is
    already authored.

    A revolute joint's position is defined entirely by ``localRot1`` when
    ``localRot0`` is the identity (true for every SimReady joint seen so far
    in this project): ``worldRot(body1) = worldRot(body0) * Rot(axis,
    position)``, which only holds when ``localRot1 == Rot(axis, -position)``.
    So baking a target angle is just overwriting ``localRot1`` with that
    quaternion before welding -- no live simulation step needed, which
    matters because every structural USD edit in this module (weld included)
    has to happen before the first ``sim.reset()`` to play correctly with
    ``GridCloner`` and the GPU pipeline. Confirmed empirically against the
    live tensor-API-commanded sweep this was derived from: baking -127.5 deg
    this way reproduces the exact pose live-commanding ``jointPosition =
    -127.5 deg`` renders.

    Only axis-X joints are needed/tested so far (every SimReady hinge prop
    encountered uses axis X) -- extend the axis handling here if a Y/Z-axis
    joint ever needs this.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(joint_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"weld_joint_at_angle: prim not found: {joint_prim_path}")
    revolute = UsdPhysics.RevoluteJoint(prim)
    axis = revolute.GetAxisAttr().Get()
    if axis != "X":
        raise NotImplementedError(f"weld_joint_at_angle: only axis X is supported, got {axis!r}")
    half = -math.radians(angle_deg) / 2.0
    new_local_rot1 = Gf.Quatf(math.cos(half), Gf.Vec3f(math.sin(half), 0.0, 0.0))
    revolute.GetLocalRot1Attr().Set(new_local_rot1)
    weld_joint(joint_prim_path)


def add_joint_drive(
    joint_prim_path: str,
    stiffness: float,
    damping: float,
    target_position_deg: float,
    drive_type: str = "angular",
) -> None:
    """Author a ``UsdPhysics.DriveAPI`` on a joint that doesn't have one.

    A joint with no authored drive at all (checked directly against the
    source USD, not assumed) has nothing for the tensor API's own
    ``set_gains``/``set_joint_position_targets`` to tune -- those only
    adjust an EXISTING drive's parameters; they don't create one. Confirmed
    empirically: calling them on such a joint is silently a no-op, and the
    joint just swings freely under gravity to its limit. Must be authored
    before the first ``sim.reset()``, same timing as every other structural
    USD edit in this module. ``target_position_deg`` matches
    ``UsdPhysics.RevoluteJoint``'s own limit convention (degrees), not the
    tensor API's radians.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(joint_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"add_joint_drive: prim not found: {joint_prim_path}")
    drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetPositionAttr(float(target_position_deg))


def add_articulation_root(prim_path: str) -> None:
    """Apply ``ArticulationRootAPI`` to a prim that doesn't already have one.

    Some vendored assets (e.g. NVIDIA SimReady props) ship a rigid body +
    internal joint pair with no articulation root at all -- fine for a
    passive prop, but the PhysX tensor API (``ArticulationView``, needed to
    read/drive the joint) only exists for prims PhysX recognizes as an
    articulation. Must be authored before the first ``sim.reset()``, same
    timing constraint as every other structural USD edit in this module.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"add_articulation_root: prim not found: {prim_path}")
    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(prim)


def normalize_joint_drives(root_prim_path: str) -> list:
    """Re-author joint drives that were applied under the wrong schema instance.

    ``UsdPhysics.DriveAPI`` is a multiple-apply schema, and for a joint PhysX
    only honours the instance named after the drive's *kind* -- ``angular`` for
    a revolute joint, ``linear`` for a prismatic one. ``omni.importer.mjcf``
    can instead name it after the joint's AXIS: dynamixel_2r's ``R1``/``R2``
    ship ``PhysicsDriveAPI:X`` with ``physics:axis = X``, which looks correct
    in usdview and is silently ignored by the solver. The joint then has no
    effective drive at all, so ``set_gains`` and
    ``set_joint_position_targets`` are both no-ops (see ``add_joint_drive``'s
    own note) and the arm cannot be commanded -- reported as simply "the arm
    doesn't move".

    Copies every parameter across rather than substituting a guess, so the
    asset author's intended gains survive. Returns the joints it fixed. Must
    run before the first ``sim.reset()``, same timing as the rest of this
    module.
    """
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"normalize_joint_drives: prim not found: {root_prim_path}")
    fixed = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            wanted = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            wanted = "linear"
        else:
            continue
        if UsdPhysics.DriveAPI(prim, wanted).GetStiffnessAttr():
            continue
        source = None
        for name in ("X", "Y", "Z", "angular", "linear", "transX", "transY", "transZ",
                     "rotX", "rotY", "rotZ"):
            if name == wanted:
                continue
            candidate = UsdPhysics.DriveAPI(prim, name)
            if candidate.GetStiffnessAttr():
                source = candidate
                break
        if source is None:
            continue
        target = UsdPhysics.DriveAPI.Apply(prim, wanted)
        for getter, creator in (
            (source.GetTypeAttr, target.CreateTypeAttr),
            (source.GetStiffnessAttr, target.CreateStiffnessAttr),
            (source.GetDampingAttr, target.CreateDampingAttr),
            (source.GetMaxForceAttr, target.CreateMaxForceAttr),
            (source.GetTargetPositionAttr, target.CreateTargetPositionAttr),
            (source.GetTargetVelocityAttr, target.CreateTargetVelocityAttr),
        ):
            attr = getter()
            if attr and attr.IsValid() and attr.Get() is not None:
                creator(attr.Get())
        fixed.append(prim.GetName())
    if fixed:
        logging.info(
            "aetos_multiam.usd_fixups: re-authored %d joint drive(s) under the instance "
            "PhysX actually reads: %s", len(fixed), ", ".join(fixed),
        )
    return fixed


def set_joint_local_pose(joint_prim_path: str, position, orientation) -> None:
    """Re-author a joint's ``localPos0``/``localRot0`` -- where its child body
    attaches to its parent.

    For a fixed joint welding a pre-baked arm onto a drone body, this is how
    the arm gets re-mounted without touching the vendored asset:
    quad_scorpion.usd mounts its arm UNDER the airframe, while this project's
    own MJCF bundle for the same robot (ambench_quad_600_top_arm_final) mounts
    it on top. ``orientation`` is (w, x, y, z).
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(joint_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"set_joint_local_pose: prim not found: {joint_prim_path}")
    joint = UsdPhysics.Joint(prim)
    if not joint:
        raise RuntimeError(f"set_joint_local_pose: not a joint: {joint_prim_path}")
    w, x, y, z = (float(v) for v in orientation)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in position)))
    joint.CreateLocalRot0Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def make_non_instanceable(prim_path: str) -> int:
    """Clear ``instanceable`` on a prim and everything under it, returning how
    many were cleared.

    PhysX will not build a rigid body out of an instanceable prim: its shapes
    live in a shared prototype, so there is nothing per-instance to simulate.
    The symptom is the tensor API reporting "did not match any rigid bodies"
    for a prim that plainly has ``PhysicsRigidBodyAPI`` on it and nothing else
    wrong -- NVIDIA's SimReady props are authored this way (their collider
    mesh sits inside the prototype and never shows up in an ordinary stage
    traverse either).
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"make_non_instanceable: prim not found: {prim_path}")
    cleared = 0
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if p.IsInstanceable():
            p.SetInstanceable(False)
            cleared += 1
    return cleared


def strip_articulation_root(prim_path: str) -> None:
    """Remove ArticulationRootAPI from a prim that's about to be made
    kinematic. PhysX just ignores (with a warning) an ArticulationRootAPI
    on a kinematic body -- but the tensor API then fails to register the
    body as queryable at all ("did not match any rigid bodies"), so the
    schema has to actually come off, not just be ignored by the solver.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"strip_articulation_root: prim not found: {prim_path}")
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


# The two materials omnidrones_340's own asset binds to its frame and its
# propellers. Reused verbatim (same MDL, same subidentifier) rather than
# approximated with an OmniPBR colour, so every drone ends up the exact black
# and red that asset already renders.
PLASTIC_MDL = (
    "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/"
    "vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl"
)
FRAME_MATERIAL = "plastic_black"
PROP_MATERIAL = "plastic_red"

_PROP_NAME_TOKENS = ("rotor", "prop")


def _define_mdl_material(stage, prim_path: str, subidentifier: str):
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
    shader.SetSourceAsset(PLASTIC_MDL, "mdl")
    shader.SetSourceAssetSubIdentifier(subidentifier, "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def apply_drone_livery(drone_prim_path: str) -> int:
    """Re-bind a drone's airframe to the shared black frame / red propeller
    livery, returning how many visual prims were re-bound.

    Only the drone itself is touched: ``base_link`` (AM-bench's assets keep
    their propellers inside its subtree rather than as separate bodies) plus
    any sibling body whose name says rotor/propeller. A mounted manipulator
    or gripper -- and a bundled-in arm, whose bodies are flat siblings of
    base_link on amov_drone_arm -- keeps whatever colours its own asset ships.

    shortcut: a prim whose *name* contains "prop" is taken to be a propeller,
    which would miscolour ambench_750's "v2_prop_arm_*" frame booms; that
    asset keeps ``DroneSpec.livery = False`` anyway. Revisit if a new asset
    names its booms that way.
    """
    stage = get_current_stage()
    root = stage.GetPrimAtPath(drone_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"apply_drone_livery: prim not found: {drone_prim_path}")

    frame = _define_mdl_material(stage, f"{drone_prim_path}/Looks/livery_frame", FRAME_MATERIAL)
    prop = _define_mdl_material(stage, f"{drone_prim_path}/Looks/livery_prop", PROP_MATERIAL)

    bodies = [stage.GetPrimAtPath(f"{drone_prim_path}/base_link")]
    bodies += [
        child for child in root.GetChildren()
        if any(token in child.GetName().lower() for token in _PROP_NAME_TOKENS)
    ]

    bound = 0
    for body in bodies:
        if not body.IsValid():
            continue
        for prim in Usd.PrimRange(body):
            # AM-bench's assets bind one material on the body prim itself as
            # "strongerThanDescendants", which silently outranks anything
            # bound per-mesh underneath -- the meshes end up re-bound and
            # still render in the asset's own colour.
            binding = prim.GetRelationship("material:binding")
            if binding and binding.GetTargets():
                UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
                    binding, UsdShade.Tokens.weakerThanDescendants
                )
            if not prim.IsA(UsdGeom.Gprim):
                continue
            relative = prim.GetPath().pathString[len(drone_prim_path):].lower()
            # "sites" are the MJCF marker gprims these assets carry (IMU, per-
            # rotor thrust points) -- not geometry anyone sees.
            if "collision" in relative or "/sites/" in relative:
                continue
            is_prop = any(token in relative for token in _PROP_NAME_TOKENS)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(prop if is_prop else frame)
            bound += 1
    logging.info("apply_drone_livery: re-bound %d visual prims under %s", bound, drone_prim_path)
    return bound
