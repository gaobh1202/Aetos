"""Spawn-time mounting: the mechanism that makes drone x arm x gripper swappable.

robosuite merges robot + gripper by splicing MJCF XML trees before ``MjSim``
is built (``RobotModel.add_gripper`` -> ``MujocoXML.merge``). There's no XML
tree in Isaac Sim -- USD referencing + a PhysX fixed joint play that role
instead: reference the child's USD, then weld the child's root link to the
parent body so the whole thing simulates as one PhysX articulation.

Used twice per ``AerialManipulatorBase.spawn()`` call: drone ``base_link`` ->
manipulator root, and (only when the gripper isn't already bundled into the
manipulator's own USD) manipulator ``eef_body_name`` -> gripper root.

**The child is referenced as a SIBLING of the parent body, not nested inside
it.** An earlier version nested the child prim under the parent body prim
(``{parent_body_path}/{child_mount_name}``) -- that seemed natural (USD
parent/child transforms would "just work"), but PhysX flatly rejects any
RigidBodyAPI prim nested inside another RigidBodyAPI prim in the Xform tree
("xformstack reset ... missing when child of rigid body in hierarchy"),
and every joint inside the nested child then fails to resolve at all
("CreateJoint - no bodies defined"), silently producing an articulation
with the manipulator's joints simply missing. This was found the same way
the fin-ray finger's identical nesting bug was found earlier in this
project: by actually booting Isaac Sim and reading the PhysX error log, not
by inspecting the schema. Since siblings can't rely on USD's automatic
parent-child transform composition, this module computes the parent body's
actual WORLD pose and composes it with the requested relative offset by
hand before placing the child -- ``omni.physx.scripts.utils.createJoint()``
then derives the joint's local frames from the two prims' real world poses
regardless of where either one lives in the hierarchy, so the weld itself
doesn't care that they're siblings now instead of parent/child.

**Follow-up -- the container-offset bug.** The world pose computed above was
being passed straight to ``prim_utils.create_prim(child_prim_path, ...,
translation=, orientation=)`` where ``child_prim_path`` is a child of
``parent_prim_path`` (the drone instance's own top-level container, which
sits at the drone's spawn position -- NOT the world origin).
``create_prim``'s ``translation``/``orientation`` author a LOCAL xformOp
relative to the prim's actual USD parent, not an absolute world pose -- so
every mount silently double-applied the parent's own spawn offset. Invisible
for every manipulator mount before this (always ``translation=(0,0,0)``,
double-counted zero is still zero) until a gripper mount used a non-zero
offset. Fixed by converting ``world_transform`` into one relative to
``parent_prim_path``'s own current world transform before calling
``create_prim`` -- see the ``container_world``/``local_transform`` lines
below. This affects every future manipulator/gripper mount with a non-zero
offset, not just one case -- check this first if a mounted part ever looks
mispositioned again.
"""

import logging

import omni.isaac.core.utils.prims as prim_utils
import omni.kit.commands
import omni.physx.scripts.utils as script_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def mount_child(
    parent_prim_path: str,
    parent_body: str,
    child_usd_path: str,
    child_mount_name: str,
    child_root_body: str,
    translation=(0.0, 0.0, 0.0),
    orientation=None,
    joint_type: str = "Fixed",
    joint_axis: str = "Z",
    joint_lower_limit=None,
    joint_upper_limit=None,
) -> str:
    """Reference ``child_usd_path`` under the parent body and weld it on.

    Args:
        parent_prim_path: absolute prim path of the already-spawned parent
            (e.g. one drone instance's root, ``/World/envs/env_0/amov_950_0``).
        parent_body: body name under the parent to mount onto, e.g. ``"base_link"``.
        child_usd_path: USD file to reference in (the manipulator's or gripper's asset).
        child_mount_name: prim name for the new reference, e.g. ``"manipulator"``.
        child_root_body: name of the child's own root rigid body, used to find
            the prim that the fixed joint should weld to the parent with.
        translation / orientation: child's pose relative to the parent body.

    Returns:
        The absolute prim path of the newly-referenced child subtree, so the
        caller can re-glob ``ArticulationView``/``RigidPrimView`` paths from it.
    """
    parent_body_path = f"{parent_prim_path}/{parent_body}"
    if not prim_utils.is_prim_path_valid(parent_body_path):
        raise ValueError(
            f"Cannot mount '{child_usd_path}' onto '{parent_body_path}': "
            "parent body prim does not exist. Did the parent already spawn?"
        )

    # Sibling of parent_prim_path (NOT nested under parent_body_path, a
    # RigidBody prim -- see module docstring). Compute the absolute world
    # pose by hand since siblings don't inherit the parent body's transform
    # the way a USD child would.
    stage = prim_utils.get_current_stage()
    parent_body_prim = prim_utils.get_prim_at_path(parent_body_path)
    parent_world = UsdGeom.Xformable(parent_body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    parent_world = parent_world.RemoveScaleShear()
    rel_pos = Gf.Vec3d(*translation)
    if orientation is not None:
        w, x, y, z = orientation
        rel_rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        rel_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0.0)
    rel_transform = Gf.Matrix4d().SetRotate(rel_rot) * Gf.Matrix4d().SetTranslate(rel_pos)
    world_transform = rel_transform * parent_world
    # create_prim's translation/orientation are LOCAL to the new prim's own
    # USD parent (parent_prim_path), not the world origin -- convert the
    # absolute world_transform computed above into one relative to
    # parent_prim_path's own current world transform (see module docstring's
    # "container-offset bug" follow-up).
    container_world = UsdGeom.Xformable(
        prim_utils.get_prim_at_path(parent_prim_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    container_world = container_world.RemoveScaleShear()
    local_transform = world_transform * container_world.GetInverse()
    local_pos = local_transform.ExtractTranslation()
    local_quat = local_transform.ExtractRotationQuat()
    local_orientation = (local_quat.GetReal(), *local_quat.GetImaginary())

    child_prim_path = f"{parent_prim_path}/{child_mount_name}"
    prim_utils.create_prim(
        child_prim_path,
        usd_path=child_usd_path,
        translation=Gf.Vec3d(local_pos),
        orientation=local_orientation,
    )

    child_body_path = f"{child_prim_path}/{child_root_body}"
    parent_prim = prim_utils.get_prim_at_path(parent_body_path)
    child_prim = prim_utils.get_prim_at_path(child_body_path)
    if not (child_prim and child_prim.IsValid()):
        raise ValueError(
            f"Expected root body '{child_root_body}' directly under '{child_prim_path}' "
            "after referencing -- check the USD's default-prim merge behavior "
            "(see feedback-isaac-usd-default-prim-merge in project memory)."
        )

    # The child USD was very likely authored to also stand alone (its own
    # ArticulationRootAPI on its root body, e.g. for testing the arm by
    # itself). A fixed joint between two *separate* PhysX articulations is a
    # much looser coupling than one continuous articulation tree -- so strip
    # the child's own articulation root here, folding its joints into the
    # parent's single articulation instead of creating a second one.
    if UsdPhysics.ArticulationRootAPI(child_prim):
        child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        logging.info(
            "aetos_multiam.usd_compose: removed ArticulationRootAPI from '%s' "
            "so it merges into the parent articulation instead of forming its own.",
            child_body_path,
        )

    joint = script_utils.createJoint(stage, joint_type, parent_prim, child_prim)
    if joint_type == "Revolute":
        revolute = UsdPhysics.RevoluteJoint(joint)
        revolute.CreateAxisAttr(joint_axis)
        revolute.CreateLowerLimitAttr(float(joint_lower_limit))
        revolute.CreateUpperLimitAttr(float(joint_upper_limit))
    else:
        UsdPhysics.FixedJoint(joint)

    # createJoint names the joint prim with a bare type-derived name (e.g.
    # "FixedJoint") -- two mount_child calls under different parents can
    # collide on that bare leaf name, which the PhysX tensor API rejects as a
    # "Duplicate joint name" even though the full paths differ. Rename to
    # something unique per mount and type-agnostic.
    unique_joint_path = f"{parent_prim_path}/{child_mount_name}_MountJoint"
    if joint.GetPath() != unique_joint_path:
        omni.kit.commands.execute(
            "MovePrim", path_from=joint.GetPath(), path_to=unique_joint_path
        )
        joint = prim_utils.get_prim_at_path(unique_joint_path)

    logging.info(
        "aetos_multiam.usd_compose: mounted '%s' (%s) onto '%s' via %s joint at '%s'",
        child_usd_path, child_root_body, parent_body_path, joint_type, joint.GetPath(),
    )
    return child_prim_path
