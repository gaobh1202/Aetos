"""Extract standalone Inspire Hand USD assets out of the two larger source
files (full G1 humanoid, and a Kuka-arm + hand demo scene), by copying just
the hand's own body/joint prim specs into fresh stages -- not re-authoring
geometry/physics, just pruning everything outside the hand subtree.

Approach: open the source file directly (not via reference, to get its own
true internal prim paths), Sdf.CopySpec() the needed prim specs into a new
stage, fix up the one relationship that needs it (the hand's proximal
joints' body0Rel, which pointed at the now-excluded wrist link), apply a
fresh ArticulationRootAPI to the hand's own root body, and export.

**g1_hands.usd specific wrinkle**: its visuals/collisions prims aren't
inline meshes -- they're a `reference` to a same-layer top-level prim named
like `/Flattened_Prototype_26` (an internal mesh-dedup scheme, NOT USD's
native scenegraph instancing -- `stage.GetPrototypes()` reports a
completely different, unrelated set of `/__Prototype_N` prims, and
`Stage.Flatten()` does NOT resolve these either, both confirmed by testing
directly). A naive copy of the hand's own body prims alone produced
dangling `Unresolved reference` warnings because the referenced prototype
prim was never copied. Fix: walk the raw layer (not the composed stage, so
deactivated/class-like prims aren't skipped) under every body being
extracted, collect every `Flattened_Prototype_N` reference target it uses,
and copy those prims too. kuka_w_hand.usd doesn't use this scheme at all
(its visuals/collisions are plain inline Mesh prims, confirmed separately)
so its extraction doesn't need this step.
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Sdf
import os

OUT_DIR = "/home/benhua/Aetos_MultiAM/aetos_multiam/assets/usd/inspire_hand"
os.makedirs(OUT_DIR, exist_ok=True)


def copy_prim(src_layer, dst_layer, src_path, dst_path):
    ok = Sdf.CopySpec(src_layer, Sdf.Path(src_path), dst_layer, Sdf.Path(dst_path))
    if not ok:
        raise RuntimeError(f"CopySpec failed: {src_path} -> {dst_path}")


def find_proto_refs(layer, prim_path):
    """All Flattened_Prototype_N targets referenced anywhere under prim_path, via raw layer specs."""
    found = set()

    def walk(path):
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            return
        for ref in spec.referenceList.GetAddedOrExplicitItems():
            if ref.primPath and "Flattened_Prototype" in str(ref.primPath):
                found.add(str(ref.primPath))
        for child_name in spec.nameChildren.keys():
            walk(path.AppendChild(child_name))

    walk(Sdf.Path(prim_path))
    return found


def rewrite_rel_target(prim, rel_name, old_path, new_path):
    rel = prim.GetRelationship(rel_name)
    if not rel or not rel.IsValid():
        return
    targets = rel.GetTargets()
    new_targets = []
    changed = False
    for t in targets:
        s = str(t)
        if s == old_path or s.startswith(old_path + "/"):
            new_targets.append(Sdf.Path(new_path + s[len(old_path):]))
            changed = True
        else:
            new_targets.append(t)
    if changed:
        rel.SetTargets(new_targets)


def rewrite_all_rel_prefixes(dst_stage, dst_subtree_path, old_prefix, new_prefix):
    """Every relationship anywhere under dst_subtree_path whose target starts
    with old_prefix gets that prefix swapped for new_prefix. Needed because
    Sdf.CopySpec() copies relationship target paths verbatim -- joint
    body0/body1 rels AND material:binding rels alike still point at the
    SOURCE file's own top-level container name after the copy, which no
    longer exists once that container got renamed in the destination."""
    count = 0
    # Usd.PrimRange's DEFAULT predicate skips "over"-specifier prims'
    # children entirely (confirmed: the Flattened_Prototype_N prims are
    # authored as `over`, and GetChildren()/default PrimRange both return
    # nothing for them even though GetAllChildren() finds real children) --
    # use the all-prims predicate so this sweep actually reaches in.
    for prim in Usd.PrimRange(dst_stage.GetPrimAtPath(dst_subtree_path), Usd.PrimAllPrimsPredicate):
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if not targets:
                continue
            new_targets = []
            changed = False
            for t in targets:
                s = str(t)
                if s == old_prefix or s.startswith(old_prefix + "/"):
                    new_targets.append(Sdf.Path(new_prefix + s[len(old_prefix):]))
                    changed = True
                else:
                    new_targets.append(t)
            if changed:
                rel.SetTargets(new_targets)
                count += 1
    return count


def extract_g1_hand(src_path, side, out_name, root_prefix):
    src = Usd.Stage.Open(src_path)
    src_layer = src.GetRootLayer()
    sdp = src.GetDefaultPrim()
    src_root = str(sdp.GetPath())

    out_path = os.path.join(OUT_DIR, out_name)
    dst = Usd.Stage.CreateNew(out_path)
    dst_layer = dst.GetRootLayer()
    UsdGeom.SetStageUpAxis(dst, UsdGeom.GetStageUpAxis(src))
    dst.SetMetadata("metersPerUnit", src.GetMetadata("metersPerUnit"))

    dst_root_name = out_name.replace(".usd", "")
    dst_root_path = f"/{dst_root_name}"
    dst.DefinePrim(dst_root_path, "Xform")
    dst.SetDefaultPrim(dst.GetPrimAtPath(dst_root_path))

    copy_prim(src_layer, dst_layer, f"{src_root}/Looks", f"{dst_root_path}/Looks")

    new_base_path = f"{dst_root_path}/hand_base"
    copy_prim(src_layer, dst_layer, f"{src_root}/{root_prefix}", new_base_path)

    finger_bodies = [
        f"{side}_index_proximal", f"{side}_index_intermediate",
        f"{side}_middle_proximal", f"{side}_middle_intermediate",
        f"{side}_pinky_proximal", f"{side}_pinky_intermediate",
        f"{side}_ring_proximal", f"{side}_ring_intermediate",
        f"{side}_thumb_proximal_base", f"{side}_thumb_proximal",
        f"{side}_thumb_intermediate", f"{side}_thumb_distal",
    ]
    for name in finger_bodies:
        copy_prim(src_layer, dst_layer, f"{src_root}/{name}", f"{dst_root_path}/{name}")

    finger_joints = [
        f"{side}_index_proximal_joint", f"{side}_index_intermediate_joint",
        f"{side}_middle_proximal_joint", f"{side}_middle_intermediate_joint",
        f"{side}_pinky_proximal_joint", f"{side}_pinky_intermediate_joint",
        f"{side}_ring_proximal_joint", f"{side}_ring_intermediate_joint",
        f"{side}_thumb_proximal_yaw_joint", f"{side}_thumb_proximal_pitch_joint",
        f"{side}_thumb_intermediate_joint", f"{side}_thumb_distal_joint",
    ]
    joints_scope_path = f"{dst_root_path}/joints"
    dst.DefinePrim(joints_scope_path, "Scope")
    for name in finger_joints:
        copy_prim(src_layer, dst_layer, f"{src_root}/joints/{name}", f"{joints_scope_path}/{name}")

    # Prototype meshes: discover every Flattened_Prototype_N the copied
    # bodies reference, then copy those prims in too (see module docstring).
    protos = set()
    protos |= find_proto_refs(src_layer, f"{src_root}/{root_prefix}")
    for name in finger_bodies:
        protos |= find_proto_refs(src_layer, f"{src_root}/{name}")
    for p in protos:
        copy_prim(src_layer, dst_layer, p, p)  # same path at top level in dst too

    # The prototypes themselves (top-level, NOT nested under dst_root_path)
    # carry their own material:binding rels at the OLD source container
    # prefix too -- the main sweep below only covers dst_root_path's own
    # subtree, so each prototype needs the same fix applied separately.
    for p in protos:
        rewrite_all_rel_prefixes(dst, p, src_root, dst_root_path)

    # Every relationship Sdf.CopySpec just carried over verbatim (joint
    # body0/body1 rels, material:binding) still points at the SOURCE
    # container's own absolute path -- fix the wrist-specific rename first
    # (exact match, so the general prefix sweep below doesn't also need to
    # touch it), then sweep everything else from src_root -> dst_root_path.
    old_root = f"{src_root}/{root_prefix}"
    for name in finger_joints:
        jprim = dst.GetPrimAtPath(f"{joints_scope_path}/{name}")
        rewrite_rel_target(jprim, "physics:body0", old_root, new_base_path)
    n_fixed = rewrite_all_rel_prefixes(dst, dst_root_path, src_root, dst_root_path)
    print(f"  rewrote {n_fixed} relationship(s) from the source container prefix", flush=True)

    base_prim = dst.GetPrimAtPath(new_base_path)
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)

    # hand_base's visuals/collisions prototypes each carry TWO meshes: the
    # real Inspire Hand base (`{side}_hand_base_link`, ~88k points) AND the
    # Unitree wrist servo's own adapter-flange geometry
    # (`{root_prefix}/mesh`, ~20k points, small compact bbox) -- the latter
    # is G1-specific mounting hardware, not part of the hand itself, and
    # the user asked for it removed (found live: it's a *separate* Mesh
    # prim under the SAME referenced prototype, not merged into one mesh,
    # so a plain RemovePrim on it is enough -- doesn't touch hand_base's
    # own authored mass/CoM/inertia, which are explicit attributes, not
    # derived from the collider geometry).
    for sub in ("visuals", "collisions"):
        sub_prim = dst.GetPrimAtPath(f"{new_base_path}/{sub}")
        ref = sub_prim.GetPrimStack()[0].referenceList.GetAddedOrExplicitItems()[0]
        proto_path = str(ref.primPath)
        flange_path = f"{proto_path}/{root_prefix}"
        if dst.GetPrimAtPath(flange_path).IsValid():
            dst.RemovePrim(flange_path)
            print(f"  removed wrist-flange mesh at {flange_path}", flush=True)

    dst_layer.Save()
    print(f"Wrote {out_path} ({len(protos)} prototype meshes copied)", flush=True)
    return out_path


def extract_kuka_hand(src_path, out_name):
    src = Usd.Stage.Open(src_path)
    src_layer = src.GetRootLayer()
    sdp = src.GetDefaultPrim()
    src_root = str(sdp.GetPath())

    out_path = os.path.join(OUT_DIR, out_name)
    dst = Usd.Stage.CreateNew(out_path)
    dst_layer = dst.GetRootLayer()
    UsdGeom.SetStageUpAxis(dst, UsdGeom.GetStageUpAxis(src))
    dst.SetMetadata("metersPerUnit", src.GetMetadata("metersPerUnit"))

    dst_root_name = out_name.replace(".usd", "")
    dst_root_path = f"/{dst_root_name}"
    dst.DefinePrim(dst_root_path, "Xform")
    dst.SetDefaultPrim(dst.GetPrimAtPath(dst_root_path))

    copy_prim(src_layer, dst_layer, f"{src_root}/Looks", f"{dst_root_path}/Looks")

    new_base_path = f"{dst_root_path}/hand_base"
    copy_prim(src_layer, dst_layer, f"{src_root}/robot1_gripper_right_hand_link", new_base_path)

    finger_bodies = [
        "robot1_gripper_right_hand_ee_link",
        "robot1_gripper_right_hand_index_bend_link",
        "robot1_gripper_right_hand_index_rota_link1",
        "robot1_gripper_right_hand_index_rota_link2",
        "robot1_gripper_right_hand_index_rota_tip",
        "robot1_gripper_right_hand_mid_link1",
        "robot1_gripper_right_hand_mid_link2",
        "robot1_gripper_right_hand_mid_tip",
        "robot1_gripper_right_hand_pinky_link1",
        "robot1_gripper_right_hand_pinky_link2",
        "robot1_gripper_right_hand_pinky_tip",
        "robot1_gripper_right_hand_ring_link1",
        "robot1_gripper_right_hand_ring_link2",
        "robot1_gripper_right_hand_ring_tip",
        "robot1_gripper_right_hand_thumb_bend_link",
        "robot1_gripper_right_hand_thumb_rota_link1",
        "robot1_gripper_right_hand_thumb_rota_link2",
        "robot1_gripper_right_hand_thumb_rota_tip",
    ]
    for name in finger_bodies:
        copy_prim(src_layer, dst_layer, f"{src_root}/{name}", f"{dst_root_path}/{name}")

    # Same relationship-rewrite need as the g1 extraction: the root body got
    # renamed (robot1_gripper_right_hand_link -> hand_base) and the whole
    # container got renamed too, and CopySpec carries every relationship
    # target over as a literal, un-remapped absolute path.
    old_root = f"{src_root}/robot1_gripper_right_hand_link"
    for prim in Usd.PrimRange(dst.GetPrimAtPath(new_base_path)):
        rewrite_rel_target(prim, "physics:body0", old_root, new_base_path)
        rewrite_rel_target(prim, "physics:body1", old_root, new_base_path)
    n_fixed = rewrite_all_rel_prefixes(dst, dst_root_path, src_root, dst_root_path)
    print(f"  rewrote {n_fixed} relationship(s) from the source container prefix", flush=True)

    base_prim = dst.GetPrimAtPath(new_base_path)
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)

    dst_layer.Save()
    print(f"Wrote {out_path}", flush=True)
    return out_path


G1_SRC = "/home/benhua/Aetos_MultiAM/aetos_multiam/assets/usd/inspire_hand_candidates/g1_hands.usd"
KUKA_SRC = "/home/benhua/Aetos_MultiAM/aetos_multiam/assets/usd/inspire_hand_candidates/kuka_w_hand.usd"

extract_g1_hand(G1_SRC, "L", "inspire_hand_g1dfq_left.usd", "left_wrist_yaw_link")
extract_g1_hand(G1_SRC, "R", "inspire_hand_g1dfq_right.usd", "right_wrist_yaw_link")
extract_kuka_hand(KUKA_SRC, "inspire_hand_kuka_right.usd")

app.close()
