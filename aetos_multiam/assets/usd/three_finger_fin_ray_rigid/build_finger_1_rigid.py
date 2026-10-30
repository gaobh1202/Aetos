"""
Build the rigid Fin-Ray finger_1 approximation (see DERIVATION.md in this
directory) as a real PhysX articulation in Isaac Sim, and export it to USD.

This is the second representation of the same structure already validated in
MuJoCo as `finger_rigid.xml` -- same 3-segment pseudo-rigid-body
discretization, same per-segment length/width/mass/stiffness numbers from
DERIVATION.md. The point of authoring it independently here (rather than
converting the MJCF file with `omni.importer.mjcf`) is that MJCF's native
body-in-body nesting and radian-based spring-hinge semantics don't carry
over 1:1 to USD Physics -- three real authoring pitfalls had to be found and
fixed by actually running this in Isaac Sim, not just writing the schema
calls and assuming they'd work:

1. PhysX rejects RigidBodyAPI prims NESTED inside another RigidBodyAPI prim
   ("xformstack reset ... missing when child of rigid body in hierarchy").
   MJCF's body-in-body tree has to become FLAT SIBLINGS here, each body
   given its own absolute world transform, connected only by joint prims
   (body0Rel/body1Rel) -- never by USD parent/child nesting.

2. PhysX articulations reject a KINEMATIC body anywhere in the chain
   ("Articulations with kinematic bodies are not supported"). A fixed-base
   link (standing in for the real gripper housing this finger eventually
   mounts to) must be a normal DYNAMIC RigidBody welded to world with a
   UsdPhysics.FixedJoint -- the same pattern Isaac Sim's own URDF importer
   uses for "Fix Base Link", not a kinematic flag.

3. The one that cost the most debugging time: UsdPhysics.Joint's
   LocalRot0/LocalRot1 attributes do NOT default to identity when left
   unauthored -- they silently produce a 0-DOF/rigid joint instead (every
   joint reads back exactly 0 position/velocity under gravity, forever,
   with no error). They must always be authored explicitly, AND when the
   two bodies either side of a joint have different world orientations
   (finger_1 is identity, finger_1_flex_0 carries the real 23.131deg tilt
   from DERIVATION.md), LocalRot0 (body0's side) must carry that relative
   rotation -- identity on both sides is only correct when both bodies
   already share the same world orientation, which is true for
   flex_hinge_1/flex_hinge_2 here but NOT for flex_hinge_0.

Validated (this file's __main__ block, run via the omnidrone conda env
which has ISAACSIM_PATH set up): builds without PhysX errors; settles under
gravity to small, physically plausible per-joint sag (order ~1deg, matching
a hand estimate from the segment masses/lever arms); and under a 0.3N
lateral tip load (mirroring the MuJoCo DERIVATION.md validation) curls
mostly at the tip joint (+2.18deg) while the base joints barely move,
reproducing Fin-Ray's known progressive/tip-first compliance.

Not yet done: finger_2/finger_3, full 3-finger reassembly, mounting this
under the real gripper housing (gripper_base) instead of a standalone world
anchor.
"""
import os

import numpy as np


def build_finger_1_rigid(stage, root_path, mount_world_pos=(0.0, -0.0315, 0.095)):
    """Author the finger_1 rigid articulation under `root_path` on `stage`.

    `mount_world_pos` is where the actuated base hinge sits in whatever
    frame `root_path` lives in (world frame if root_path is directly under
    /World; the gripper housing's frame once this is mounted for real).
    Returns the root prim path (for ArticulationView/mounting elsewhere).
    """
    from pxr import UsdGeom, UsdPhysics, Sdf, Gf
    import omni.isaac.core.utils.prims as prim_utils

    ROOT = root_path
    prim_utils.create_prim(ROOT, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(ROOT))
    JOINTS = f"{ROOT}/joints"
    prim_utils.create_prim(JOINTS, "Scope")

    TILT_DEG = 23.131  # DERIVATION.md section 2
    tilt_rad = np.radians(TILT_DEG)
    cos_t, sin_t = np.cos(tilt_rad), np.sin(tilt_rad)

    def rot_x(v):
        x, y, z = v
        return np.array([x, y * cos_t - z * sin_t, y * sin_t + z * cos_t])

    mount_pos = np.array(mount_world_pos)
    flex0_local_in_mount = np.array([0.0, -0.02504, 0.01316])
    seg_len = 0.034416

    flex0_pos = mount_pos + flex0_local_in_mount
    flex1_pos = flex0_pos + rot_x([0, 0, seg_len])
    flex2_pos = flex1_pos + rot_x([0, 0, seg_len])

    bodies = {
        "finger_1": dict(pos=mount_pos, tilt_deg=0.0, kind="mount"),
        "finger_1_flex_0": dict(pos=flex0_pos, tilt_deg=TILT_DEG,
                                 width=0.024, mass=0.004329, k=0.09372, c=0.0010),
        "finger_1_flex_1": dict(pos=flex1_pos, tilt_deg=TILT_DEG,
                                 width=0.01620, mass=0.0029224, k=0.02884, c=0.0008),
        "finger_1_flex_2": dict(pos=flex2_pos, tilt_deg=TILT_DEG,
                                 width=0.00856, mass=0.0015444, k=0.00426, c=0.0005),
    }

    for name, b in bodies.items():
        path = f"{ROOT}/{name}"
        prim_utils.create_prim(path, "Xform", translation=Gf.Vec3f(*b["pos"].tolist()))
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        if b["tilt_deg"] != 0.0:
            xformable.AddRotateXOp().Set(b["tilt_deg"])

        if b.get("kind") == "mount":
            geom = UsdGeom.Cube.Define(stage, path + "/geom")
            geom.GetSizeAttr().Set(0.02)
            geom.AddTranslateOp().Set(Gf.Vec3f(0, -0.0154, -0.0022))
            mass_val = 0.025
        else:
            geom = UsdGeom.Cube.Define(stage, path + "/geom")
            geom.GetSizeAttr().Set(1.0)
            geom.AddScaleOp().Set(Gf.Vec3f(0.014, b["width"], seg_len))
            geom.AddTranslateOp().Set(Gf.Vec3f(0, 0, seg_len / 2.0))
            mass_val = b["mass"]

        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        body_prim = stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        mass_api.GetMassAttr().Set(mass_val)
        if "width" in b:
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0, 0, seg_len / 2.0))

    def make_hinge(joint_name, body0_path, body1_path, local_pos0, lower_deg, upper_deg,
                   k_rad=None, c_rad=None, rot0_x_deg=0.0):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{JOINTS}/{joint_name}")
        joint.CreateAxisAttr("X")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        half = np.radians(rot0_x_deg) / 2.0
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(np.cos(half)), float(np.sin(half)), 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLowerLimitAttr().Set(lower_deg)
        joint.CreateUpperLimitAttr().Set(upper_deg)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        if k_rad is not None:
            # PhysX angular-drive stiffness/damping are per-DEGREE here
            # (limits are authored in degrees), unlike MJCF's per-radian.
            drive.CreateStiffnessAttr(float(k_rad * np.pi / 180.0))
            drive.CreateDampingAttr(float(c_rad * np.pi / 180.0))
        else:
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)

    world_anchor = UsdPhysics.FixedJoint.Define(stage, f"{JOINTS}/mount_to_world")
    world_anchor.CreateBody1Rel().SetTargets([Sdf.Path(f"{ROOT}/finger_1")])
    world_anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(*mount_pos.tolist()))
    world_anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    world_anchor.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    world_anchor.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    make_hinge("flex_hinge_0", f"{ROOT}/finger_1", f"{ROOT}/finger_1_flex_0",
               tuple(flex0_local_in_mount.tolist()), 0.0, 60.0, 0.09372, 0.0010,
               rot0_x_deg=TILT_DEG)
    make_hinge("flex_hinge_1", f"{ROOT}/finger_1_flex_0", f"{ROOT}/finger_1_flex_1",
               (0.0, 0.0, seg_len), 0.0, 60.0, 0.02884, 0.0008)
    make_hinge("flex_hinge_2", f"{ROOT}/finger_1_flex_1", f"{ROOT}/finger_1_flex_2",
               (0.0, 0.0, seg_len), 0.0, 60.0, 0.00426, 0.0005)

    return ROOT


if __name__ == "__main__":
    import os
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    import omni.usd
    from omni.isaac.core.world import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    # Build at a top-level path (not nested under /World): the exported
    # file's defaultPrim must be a root-level prim for
    # add_reference_to_stage() to resolve it later as a reusable asset.
    root = build_finger_1_rigid(stage, "/finger_1_rigid")

    world.reset()
    art = Articulation(prim_path=root)
    art.initialize()
    print("dof_names:", art.dof_names, "num_dof:", art.num_dof, flush=True)

    rp_tip = RigidPrim(prim_path=f"{root}/finger_1_flex_2")

    for step in range(180):
        world.step(render=False)
    rest_pos = np.degrees(art.get_joint_positions()).copy()
    print("REST (gravity-only) joint_pos(deg):", rest_pos, flush=True)

    from omni.isaac.dynamic_control import _dynamic_control as dc
    dc_iface = dc.acquire_dynamic_control_interface()
    tip_rb = dc_iface.get_rigid_body(f"{root}/finger_1_flex_2")
    tip_world_pos, _ = rp_tip.get_world_pose()
    force = (0.0, 0.3, 0.0)
    for step in range(180):
        dc_iface.apply_body_force(tip_rb, force, tuple(tip_world_pos.tolist()), True)
        world.step(render=False)
    loaded_pos = np.degrees(art.get_joint_positions()).copy()
    print("LOADED (0.3N @ tip) joint_pos(deg):", loaded_pos, flush=True)
    print("DELTA (loaded - rest):", loaded_pos - rest_pos, flush=True)

    stage.SetDefaultPrim(stage.GetPrimAtPath(root))
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "meshes", "usd",
        "three_finger_fin_ray_rigid", "finger_1_rigid.usd",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stage.GetRootLayer().Export(out_path)
    print("Exported USD to", out_path, flush=True)

    simulation_app.close()
