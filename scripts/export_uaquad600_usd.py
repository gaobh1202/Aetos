"""Bake uaquad600_ambencharm_top into one standalone USD -- WITH the arm
re-mount surgery AM-bench's own quad_scorpion.usd never actually got.

Background: aetos_multiam/robots/manipulators/ambench_quad_arm.py's
docstring already documents a "top-mounted" target pose for the arm
(mount at body-frame (0.1, 0, 0.003) m, quat (0,1,0,0) i.e. flipped 180 deg
about X), cross-checked against assets/mjcf/ambench_quad_600_top_arm_final/
robot.xml -- but that flip was only ever applied to the MJCF viewer-export
copy, not to the actual Isaac Sim USD
(/home/benhua/ambench/source/ambench/ambench/assets/robots/quad_scorpion.usd)
that ``ambench_600``'s DroneSpec points at. Confirmed by direct inspection:
the live USD's ``arm_base_link`` sits at world (0.1, 0, -0.053) with
identity rotation -- i.e. still AM-bench's original BELOW-the-drone mount,
not the documented top-mounted one. Since ``bundled_into_drone=True`` means
AerialManipulatorBase never calls mount_child() for this preset (arm+drone
are one pre-baked articulation, not separately mountable), there is no
mount_translation/mount_orientation knob to flip -- the only way to
actually get a top-mounted uaquad600 is to edit the arm's body poses
in-place, which is what this script does, entirely on an in-memory
referenced copy of the stage (AM-bench's own source file on disk is never
touched).

The arm+gripper chain (8 rigid bodies, confirmed via direct joint
introspection of quad_scorpion.usd's /quad_scorpion_urdf/joints/* --
see CONVERSION_NOTES for the equivalent MJCF breakdown):
    arm_base_link -> arm_link1_yaw_link -> arm_link2_pitch_link ->
    arm_link3_pitch_link -> arm_link4_pitch_link -> ee_link ->
    manipulation_lfinger_link / manipulation_rfinger_link
All 8 are flat siblings directly under the drone's top-level prim (not
USD-nested under arm_base_link), each with its own absolute-looking
pre-baked rest-pose xformOp -- same "flat sibling + full joint graph"
authoring convention seen throughout this project's other MJCF-derived
assets. Flattening only arm_base_link's own joint would leave the other 7
bodies' *static* xformOps stuck in the old bottom-mounted geometry (no
physics solve runs during a spawn()-only + Flatten() export to fix that
up), so every body in the chain gets the SAME rigid transform applied: the
one that carries the OLD arm_base_link pose to the NEW (documented) one,
exactly as if the whole arm+gripper assembly were unbolted as one rigid
group and rebolted at the new mount point/orientation. The
``arm_base_fixed`` PhysicsFixedJoint's own local frame (on body0=base_link)
is updated to match, so the authored rest pose and the joint constraint
stay consistent (the dynamixel_2r saga earlier this project found that
exact kind of mismatch to be a real source of simulation-time pose snaps --
see project memory -- so this is not just cosmetic).
"""
import logging
import os

from aetos_multiam import init_simulation_app

simulation_app = init_simulation_app({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from aetos_multiam.robots.drones.ambench_600 import AMBENCH_600_USD_PATH

logging.basicConfig(level=logging.INFO)

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1 / 100,
    rendering_dt=1 / 100,
    backend="torch",
    physics_prim_path="/physicsScene",
    device="cpu",
    sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
)

ROOT_PATH = "/uaquad600_ambencharm_top"
prim_utils.create_prim(ROOT_PATH, usd_path=AMBENCH_600_USD_PATH)

stage = prim_utils.get_current_stage()
cache = UsdGeom.XformCache()

ARM_CHAIN = [
    "arm_base_link",
    "arm_link1_yaw_link",
    "arm_link2_pitch_link",
    "arm_link3_pitch_link",
    "arm_link4_pitch_link",
    "ee_link",
    "manipulation_lfinger_link",
    "manipulation_rfinger_link",
]

# Target pose documented in ambench_quad_arm.py / robot.xml: mount at
# (0.1, 0, 0.003) m, flipped 180 deg about X (quat wxyz = 0,1,0,0).
TARGET_POS = Gf.Vec3d(0.1, 0.0, 0.003)
TARGET_QUAT = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))

arm_base_prim = stage.GetPrimAtPath(f"{ROOT_PATH}/arm_base_link")
old_world = cache.GetLocalToWorldTransform(arm_base_prim)
old_world = old_world.RemoveScaleShear()

new_world = Gf.Matrix4d().SetRotate(Gf.Rotation(TARGET_QUAT)) * Gf.Matrix4d().SetTranslate(TARGET_POS)

# T carries any point from the OLD arm_base_link frame to the NEW one:
# world_new = T * world_old  =>  T = new_world * old_world^-1
T = old_world.GetInverse() * new_world

logging.info("Old arm_base_link world pose: pos=%s", old_world.ExtractTranslation())
logging.info("New arm_base_link world pose: pos=%s quat=%s", TARGET_POS, TARGET_QUAT)

for name in ARM_CHAIN:
    prim = stage.GetPrimAtPath(f"{ROOT_PATH}/{name}")
    if not prim.IsValid():
        raise RuntimeError(f"Expected body '{name}' not found under {ROOT_PATH}")
    old_body_world = cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
    new_body_world = old_body_world * T
    new_pos = new_body_world.ExtractTranslation()
    new_rot = new_body_world.ExtractRotationQuat()

    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(new_pos)
    if "xformOp:orient" in ops:
        ops["xformOp:orient"].Set(new_rot)
    logging.info("Repose %s: pos=%s", name, new_pos)

# Keep the arm_base_fixed joint's authored local frame consistent with the
# body poses just written (localPos1/localRot1, on body1=arm_base_link,
# were already (0,0,0)/identity and stay that way -- the joint frame
# coincides with arm_base_link's own origin, same convention as before).
joint = stage.GetPrimAtPath(f"{ROOT_PATH}/joints/arm_base_fixed")
joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(TARGET_POS))
joint.GetAttribute("physics:localRot0").Set(Gf.Quatf(TARGET_QUAT))
logging.info("Updated arm_base_fixed joint localPos0/localRot0 to match.")

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "aetos_multiam", "assets",
    "meshes", "usd", "combined",
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "uaquad600_ambencharm_top.usd")

flat_layer = stage.Flatten(addSourceFileComment=False)
flat_layer.Export(out_path)

logging.info("Exported flattened USD to %s", out_path)
logging.info("Root prim: %s", ROOT_PATH)

simulation_app.close()
