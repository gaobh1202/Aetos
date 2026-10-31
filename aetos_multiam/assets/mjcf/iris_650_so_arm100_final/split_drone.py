"""Split robot.xml's combined iris_650+so_arm100 MJCF into a standalone
drone-only MJCF (``iris_650_drone.xml``) -- the ``iris_650`` body (freejoint,
frame geom, 4 rotor bodies) with the ``arm_carrier``/``Base``/... subtree
(the original SO-ARM100) removed. The arm itself is NOT re-extracted from
this MJCF: the user wants Lightwheel's leisaac so101_follower.usd substitute
mounted instead (see aetos_multiam/robots/manipulators/so_arm100.py), so
only the drone side needs converting. This bundle's own so-arm100
subtree is only used as a numeric reference for the mount transform
(see aetos_multiam/robots/manipulators/so_arm100.py's mount_orientation
derivation).

Mirrors the hummingbird_2r_three_finger_50pct/split_combo.py pattern:
re-rooted under its own <worldbody>, with only the <asset> mesh defs it
actually references, same <compiler meshdir> so mesh file= paths keep
resolving. No <default> classes are needed -- none of the drone geoms use
MJCF default classes (so_arm100's defaults are irrelevant, dropped).
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


def referenced_mesh_names(subtree):
    return {g.get("mesh") for g in subtree.iter("geom") if g.get("mesh")}


def strip_subtree(parent_body, child_name):
    for child in list(parent_body):
        if child.tag == "body" and child.get("name") == child_name:
            parent_body.remove(child)
            return
    raise KeyError(f"child body '{child_name}' not found")


def main():
    tree = ET.parse(SRC)
    src_root = tree.getroot()

    drone_body = find_body(src_root, "iris_650")
    strip_subtree(drone_body, "arm_carrier")

    meshes = referenced_mesh_names(drone_body)

    mujoco = ET.Element("mujoco", {"model": "iris_650_drone"})
    compiler = src_root.find("compiler")
    if compiler is not None:
        mujoco.append(compiler)
    option = src_root.find("option")
    if option is not None:
        mujoco.append(option)

    asset_src = src_root.find("asset")
    asset = ET.SubElement(mujoco, "asset")
    if asset_src is not None:
        for m in asset_src.findall("mesh"):
            if m.get("name") in meshes:
                asset.append(m)

    worldbody = ET.SubElement(mujoco, "worldbody")
    worldbody.append(drone_body)

    doc = ET.ElementTree(mujoco)
    ET.indent(doc, space="  ")
    out_path = HERE / "iris_650_drone.xml"
    doc.write(out_path, encoding="unicode", xml_declaration=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
