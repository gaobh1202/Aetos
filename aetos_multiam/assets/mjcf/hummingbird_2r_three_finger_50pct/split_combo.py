"""Split robot.xml's combined hummingbird+dynamixel_2r+three_finger_gripper
MJCF into two standalone MJCF files, each re-rooted under its own
<worldbody> with only the <asset> mesh/material defs it actually references:

1. ``dynamixel_2r_arm.xml`` -- the ``arm_carrier`` subtree (arm_carrier ->
   base_2 -> first_segment [joint R1] -> second_segment [joint R2] ->
   tool_adapter), with the ``gripper_base`` body (and everything under it)
   removed -- ``tool_adapter`` is kept as an empty mount point. Meant for
   Isaac Sim's own ``omni.importer.mjcf`` (plain rigid links + 2 hinges, no
   flexcomp/equality -- the two things that importer can't handle anyway,
   and both only existed inside the part we're removing).

2. ``three_finger_gripper_50pct.xml`` -- the ``gripper_base`` subtree
   (finger_1/2/3, each still a flexcomp), re-rooted the same way the
   standalone ``three_finger_gripper_final/gripper.xml`` already is
   (``<worldbody><body name="gripper_base">...``) so the existing
   ``tools/convert_model.py``/``convert_model_rigid.py`` pipeline (built
   against that standalone file) works against this one unmodified -- this
   is the SAME gripper design, just at this bundle's own 50% scale instead
   of the original full scale.

Both outputs keep the ``<compiler meshdir="...">`` pointing at this
bundle's own ``assets/`` (so mesh file= paths keep resolving), and both are
otherwise self-contained (no includes).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "robot.xml"


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body '{name}' not found")


def referenced_mesh_material_names(subtree):
    meshes, materials = set(), set()
    for g in subtree.iter("geom"):
        if g.get("mesh"):
            meshes.add(g.get("mesh"))
        if g.get("material"):
            materials.add(g.get("material"))
    return meshes, materials


def build_standalone(src_tree, body, keep_default_classes, extra_elements=()):
    src_root = src_tree.getroot()
    meshes, materials = referenced_mesh_material_names(body)

    mujoco = ET.Element("mujoco", {"model": src_root.get("model", "model") + "_part"})
    compiler = src_root.find("compiler")
    if compiler is not None:
        mujoco.append(compiler)
    default = src_root.find("default")
    if default is not None:
        new_default = ET.Element("default")
        for d in default.findall("default"):
            if d.get("class") in keep_default_classes:
                new_default.append(d)
        if len(new_default):
            mujoco.append(new_default)

    asset_src = src_root.find("asset")
    asset = ET.SubElement(mujoco, "asset")
    if asset_src is not None:
        for m in asset_src.findall("mesh"):
            implicit_name = Path(m.get("file")).stem
            mesh_name = m.get("name", implicit_name)
            if mesh_name in meshes:
                # Isaac Sim's MJCF importer (unlike real MuJoCo) does not
                # derive a mesh's name from its file= stem when name= is
                # omitted -- segfaults on a <mesh> with no explicit name.
                # Always author it explicitly in the trimmed output.
                if m.get("name") is None:
                    m.set("name", implicit_name)
                asset.append(m)
        for mat in asset_src.findall("material"):
            if mat.get("name") in materials:
                asset.append(mat)

    worldbody = ET.SubElement(mujoco, "worldbody")
    worldbody.append(body)

    for el in extra_elements:
        mujoco.append(el)

    return ET.ElementTree(mujoco)


def strip_subtree(parent_body, child_name):
    for child in list(parent_body):
        if child.tag == "body" and child.get("name") == child_name:
            parent_body.remove(child)
            return
    raise KeyError(f"child body '{child_name}' not found")


def main():
    # --- 1. arm, gripper_base removed, tool_adapter kept as empty mount point ---
    tree_arm = ET.parse(SRC)
    arm_carrier = find_body(tree_arm.getroot(), "arm_carrier")
    tool_adapter = find_body(arm_carrier, "tool_adapter")
    strip_subtree(tool_adapter, "gripper_base")
    # arm_carrier's own pos/quat (absent -> identity, same frame as the
    # drone's base_link) is preserved as-is; it's already the top body here.
    arm_doc = build_standalone(tree_arm, arm_carrier, keep_default_classes={"dynamixel_2r"})
    ET.indent(arm_doc, space="  ")
    arm_doc.write(HERE / "dynamixel_2r_arm.xml", encoding="unicode", xml_declaration=True)
    print("wrote dynamixel_2r_arm.xml", flush=True)

    # --- 2. gripper_base subtree, re-rooted (own pos/quat reset to origin,
    # matching the standalone three_finger_gripper_final/gripper.xml's own
    # root body, which also has no pos/quat -- the pos/quat gripper_base
    # carries here is purely its offset *inside this combo*, not part of
    # the gripper's own geometry, and convert_model.py's extract() never
    # reads the root body's own pos/quat anyway, only each finger's). ---
    tree_grip = ET.parse(SRC)
    arm_carrier2 = find_body(tree_grip.getroot(), "arm_carrier")
    tool_adapter2 = find_body(arm_carrier2, "tool_adapter")
    gripper_base = find_body(tool_adapter2, "gripper_base")
    if "pos" in gripper_base.attrib:
        del gripper_base.attrib["pos"]
    if "quat" in gripper_base.attrib:
        del gripper_base.attrib["quat"]
    grip_doc = build_standalone(tree_grip, gripper_base, keep_default_classes={"three_finger"})
    ET.indent(grip_doc, space="  ")
    grip_doc.write(HERE / "three_finger_gripper_50pct.xml", encoding="unicode", xml_declaration=True)
    print("wrote three_finger_gripper_50pct.xml", flush=True)


if __name__ == "__main__":
    main()
