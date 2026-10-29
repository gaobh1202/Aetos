"""Load all 5 aerial-manipulator MJCF bundles into ONE MuJoCo scene (not 5
separate renders, see 00_load_each_aerial_manipulator_mujoco.py) and
render it.

Plain `mujoco.MjModel.from_xml_path()` can't do this with raw `<include>`:
each bundle's `robot.xml` was authored standalone and reuses generic names
across different bundles (several have their own `link1`/`joint1`/
`base_link`) -- five `<include>`s in one file collide on duplicate
body/joint/mesh/material/tendon/class names at compile time.

Tried `dm_control.mjcf` first (it's built exactly for this: `attach()`
auto-prefixes a sub-model's names on merge) -- dead end. Even its newest
release (1.0.50, installed fresh in a throwaway venv since the local conda
envs only had 1.0.8) validates attributes against its own bundled schema,
which doesn't know several newer MJCF features these bundles genuinely
use and plain `mujoco` already compiles fine: `integrator="implicitfast"`,
`joint actuatorfrcrange=`, and -- the one that actually killed this
approach, in the hummingbird bundle's `flexcomp`'s own nested `<contact
internal="false">` element -- a flex-specific attribute dm_control's
schema has no entry for at all. Not a version lag fixable by upgrading;
dm_control's flex/soft-body schema coverage is just incomplete.

So: a from-scratch namespace-prefixing merger using plain
`xml.etree.ElementTree` (stdlib -- this script has no dm_control
dependency at all, runs in any env with `mujoco`+`numpy`+`imageio`),
validated the only way that matters here -- `mujoco.MjModel.from_xml_path()`
on the merged result (the same compiler that already accepts every one of
these 5 files individually, confirmed in
00_load_each_aerial_manipulator_mujoco.py). Every MJCF name a later
element can reference by string
(mesh/material/texture/body/joint/site/tendon/default-class) gets a
per-bundle prefix; every attribute that CONSUMES such a name gets the
same prefix applied to its value. Geom/camera/actuator/sensor/equality
names are prefixed too even though nothing in these 5 files happens to
reference them by name, for safety against future bundles that might.

Three other real, found-by-running-it issues along the way, all now
handled generically rather than special-cased per bundle:
1. `<compiler meshdir="...">` -- 4 of 5 bundles set this (one doesn't),
   so a mesh's `file=` value is relative to `meshdir` first, THEN the
   bundle's own directory. Missing this gave "Error opening file" with a
   plausible-looking but wrong path.
2. `<keyframe><key qpos="..." ctrl=".../></keyframe>` -- each key's
   vectors are sized for that ONE bundle's own nq/nu, wrong length for
   the merged model's much larger nq/nu. Dropped entirely (unneeded for a
   static "did it all load" viewer) rather than merged.
3. "free joint can only be used on top level" -- wrapping a free-jointed
   top-level body in a new positioning parent body (the obvious way to
   offset it) breaks this MuJoCo rule. Fix: fold the per-bundle offset
   directly into that body's own existing `pos` instead of adding a
   wrapper.

A 4th issue surfaced after the first render looked right at a glance:
amov950_airbot_final's body tree was never elevated above the ground at
all (root at z=0, airbot_play arm hanging down to z=-0.45 or lower once
posed open) -- i.e. entirely UNDER this scene's floor plane (also z=0).
An opaque floor blocks line of sight to anything beneath it from any
above-ground camera, so the arm wasn't hidden behind the drone body as
first suspected, it was hidden behind the floor itself. The other 4
bundles already sit with their root around z=1.1-1.5 (comfortably
elevated) and needed no fix. Confirmed by checking each bundle's own body
z-range directly, not by guessing from the render.

Run from anywhere (paths are resolved relative to the repo root, not the
current working directory):

    python examples/01_combine_all_aerial_manipulators_mujoco.py
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import imageio.v2 as imageio

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
BUNDLE_ROOT = os.path.join(REPO_ROOT, "aetos_multiam", "assets", "mjcf")

BUNDLES = [
    ("amov950_airbot_bottom", "amov950_airbot_final", "am"),
    ("hummingbird340_dynamixel2r_bottom", "hummingbird_2r_three_finger_50pct", "hb"),
    ("iris650_soarm100_bottom", "iris_650_so_arm100_final", "ir"),
    ("uaquad600_ambencharm_top", "ambench_quad_600_top_arm_final", "qd"),
    ("fahexa750_ambencharm_top", "ambench_fa_hexa_750_final", "hx"),
]

N = len(BUNDLES)
SPACING = 2.2  # center-to-center along a straight line, not a ring
LINE_SPAN = (N - 1) * SPACING

# z offsets per bundle: amov950_airbot_final is the one source file whose
# own body tree was never elevated above the ground plane at all -- see
# module docstring. The other 4 already sit with their root around
# z=1.1-1.5 and need no lift.
Z_LIFT = {"amov950_airbot_bottom": 1.2}
OFFSETS = [
    ((i - (N - 1) / 2) * SPACING, 0.0, Z_LIFT.get(BUNDLES[i][0], 0.0))
    for i in range(N)
]

# (tag, attribute) pairs that DEFINE a name, keyed by which "namespace" that
# name belongs to -- MJCF requires uniqueness within each kind, not globally,
# but prefixing everything the same way is simplest and still correct.
DEFINING = {
    "mesh": "name", "material": "name", "texture": "name",
    "body": "name", "joint": "name", "freejoint": "name", "site": "name", "camera": "name",
    "light": "name", "geom": "name", "default": "class",
    "flexcomp": "name", "flex": "name", "skin": "name", "numeric": "name",
    "fixed": "name", "spatial": "name",  # tendon subtypes
}
# `tag path suffix -> attribute` pairs that CONSUME (reference) a name.
# Keyed loosely by tag name since these appear under <actuator>/<sensor>/
# <equality>/<tendon>/<contact> with varying parent structure across bundles.
REFERENCING_ATTRS = [
    "mesh", "material", "texture", "joint", "joint1", "joint2",
    "body1", "body2", "site", "site1", "site2", "tendon",
    "class", "childclass", "geom1", "geom2", "objname", "target",
]


def prefix_mjcf(path, prefix):
    tree = ET.parse(path)
    root = tree.getroot()

    defined_names = set()
    for el in root.iter():
        attr = DEFINING.get(el.tag)
        if attr is not None and attr in el.attrib:
            defined_names.add(el.attrib[attr])

    for el in root.iter():
        # Rewrite this element's own defining name/class, if any.
        attr = DEFINING.get(el.tag)
        if attr is not None and attr in el.attrib:
            el.attrib[attr] = prefix + el.attrib[attr]
        # Rewrite any attribute that references a name we just collected --
        # only if the un-prefixed value actually matches a known defined
        # name (avoids mangling numeric-looking or coincidental values).
        for ref_attr in REFERENCING_ATTRS:
            if ref_attr in el.attrib and el.attrib[ref_attr] in defined_names:
                el.attrib[ref_attr] = prefix + el.attrib[ref_attr]

    # Everything directly under <actuator>/<sensor>/<equality> -- whatever
    # its specific tag (motor/position/velocity/.../jointpos/framepos/...,
    # an open-ended list this project's 5 bundles don't all share) -- also
    # needs its own `name` prefixed for global uniqueness, even though
    # nothing in any of these 5 files happens to reference such a name
    # back (DEFINING only lists tags common to all; this catches the rest).
    for section_tag in ("actuator", "sensor", "equality"):
        section = root.find(section_tag)
        if section is not None:
            for child in section:
                if "name" in child.attrib:
                    child.attrib["name"] = prefix + child.attrib["name"]

    # Mesh/texture `file=` paths are relative to THIS bundle's own
    # directory, PLUS whatever <compiler meshdir="..."> that bundle itself
    # sets (4 of these 5 bundles use meshdir="assets", one doesn't -- the
    # merged file carries no meshdir of its own, so each file= value needs
    # that prefix folded in explicitly) -- rewrite the combined result to
    # be relative to this script's directory, where the merged file lives.
    bundle_dir = os.path.dirname(path)
    rel_to_here = os.path.relpath(bundle_dir, EXAMPLES_DIR)
    compiler_el = root.find("compiler")
    meshdir = compiler_el.attrib.get("meshdir", "") if compiler_el is not None else ""
    for el in root.iter():
        if "file" in el.attrib:
            el.attrib["file"] = os.path.join(rel_to_here, meshdir, el.attrib["file"])

    return root


def merge(sections, offsets, prefixes):
    merged = ET.Element("mujoco", {"model": "all_five_combined"})
    ET.SubElement(merged, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(merged, "option", {"gravity": "0 0 0"})

    visual = ET.SubElement(merged, "visual")
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-20",
                                      "offwidth": "1920", "offheight": "1080"})
    ET.SubElement(visual, "headlight", {"diffuse": ".7 .7 .7", "ambient": ".4 .4 .4"})

    asset = ET.SubElement(merged, "asset")
    ET.SubElement(asset, "texture", {
        "name": "viewer_sky", "type": "skybox", "builtin": "gradient",
        "rgb1": ".30 .43 .57", "rgb2": ".06 .09 .14", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {
        "name": "viewer_grid", "type": "2d", "builtin": "checker",
        "rgb1": ".18 .24 .30", "rgb2": ".12 .17 .22", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {
        "name": "viewer_floor", "texture": "viewer_grid",
        "texrepeat": "20 20", "texuniform": "true", "reflectance": ".05"})

    default = ET.SubElement(merged, "default")
    worldbody = ET.SubElement(merged, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 6", "directional": "true", "dir": "0 0 -1"})
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "6 6 .01",
        "material": "viewer_floor", "contype": "0", "conaffinity": "0"})

    contact = ET.SubElement(merged, "contact")
    tendon = ET.SubElement(merged, "tendon")
    equality = ET.SubElement(merged, "equality")
    actuator = ET.SubElement(merged, "actuator")
    sensor = ET.SubElement(merged, "sensor")

    SECTION_TARGETS = {
        "asset": asset, "default": default, "contact": contact,
        "tendon": tendon, "equality": equality, "actuator": actuator,
        "sensor": sensor,
    }
    # <keyframe>'s qpos/ctrl vectors are each sized for that ONE bundle's
    # own nq/nu -- wrong length for the combined model's much larger
    # nq/nu, and not needed for a "just load and look at it" scene, so
    # keyframes are dropped entirely rather than merged.
    DROP_SECTIONS = {"keyframe"}

    for root, offset, prefix in zip(sections, offsets, prefixes):
        for child in list(root):
            if child.tag == "worldbody":
                # A body with a freejoint/free <joint type="free"> MUST be
                # a direct child of worldbody in MuJoCo ("free joint can
                # only be used on top level") -- confirmed the hard way,
                # an extra wrapper <body pos=offset> one level up breaks
                # the 2 bundles that use a free-floating drone body. So
                # the offset gets folded directly into each top-level
                # body's own existing `pos` (defaulting to the origin if
                # it didn't author one) instead of adding a new parent.
                for body_child in list(child):
                    if body_child.tag == "body":
                        existing = body_child.attrib.get("pos", "0 0 0")
                        bx, by, bz = (float(v) for v in existing.split())
                        new_pos = (bx + offset[0], by + offset[1], bz + offset[2])
                        body_child.attrib["pos"] = f"{new_pos[0]} {new_pos[1]} {new_pos[2]}"
                    worldbody.append(body_child)
            elif child.tag in SECTION_TARGETS:
                for sub in list(child):
                    SECTION_TARGETS[child.tag].append(sub)
            elif child.tag in DROP_SECTIONS:
                pass
            elif child.tag in ("compiler", "option", "visual"):
                pass  # the merged top-level ones above already cover this
            else:
                merged.append(child)  # anything unexpected: keep, top-level

    return ET.ElementTree(merged)


sections, prefixes = [], []
for preset_name, bundle_dir, prefix in BUNDLES:
    robot_path = os.path.join(BUNDLE_ROOT, bundle_dir, "robot.xml")
    sections.append(prefix_mjcf(robot_path, prefix + "_"))
    prefixes.append(prefix)
    print(f"namespaced {preset_name:38s} <- {bundle_dir:34s} prefix={prefix!r}", flush=True)

merged_tree = merge(sections, OFFSETS, prefixes)
out_xml = os.path.join(EXAMPLES_DIR, "all_five_combined.xml")
merged_tree.write(out_xml, xml_declaration=True, encoding="utf-8")
print(f"\nWrote combined MJCF: {out_xml}", flush=True)

model = mujoco.MjModel.from_xml_path(out_xml)
data = mujoco.MjData(model)

# amov950_airbot's arm (joint2/3 default near their OWN upper limit) sits
# folded up tight against the drone's belly at qpos=0 -- found by actually
# rendering it: a close-up showed only a dark lump under the body, not
# recognizable as an arm at all, unlike the other 4 bundles' arms which
# already read clearly at their own qpos=0. Posing it open/extended here
# (same numbers tried standalone first) makes it actually visible instead
# of silently correct-but-invisible.
for jname, val in [("am_joint2", -1.3), ("am_joint3", 1.6), ("am_joint5", 0.4)]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    data.qpos[model.jnt_qposadr[jid]] = val

# mj_forward only (no mj_step): this is a static "does everything load and
# sit together correctly" viewer, not a physics run -- the hummingbird
# bundle's flexcomp fingers have no passive stiffness/damping of their own
# (same "not rigid and has no equality constraints or passive forces"
# warning 00_load_each_aerial_manipulator_mujoco.py already gets
# individually) and went numerically unstable within a handful of steps
# the one time this was tried with gravity off and a few mj_step() calls;
# mj_forward alone never touches velocities so it can't diverge the same
# way, and still gives this script's `data` the correct kinematic pose to
# render from.
mujoco.mj_forward(model, data)
print(f"\nCombined model compiled+stepped OK: nq={model.nq} nbody={model.nbody} "
      f"njnt={model.njnt} nu={model.nu}", flush=True)

renderer = mujoco.Renderer(model, height=1080, width=1920)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.5]
cam.distance = LINE_SPAN * 0.95
cam.azimuth = 90
cam.elevation = -14
renderer.update_scene(data, camera=cam)
img = renderer.render()
renderer.close()

out_png = os.path.join(EXAMPLES_DIR, "all_five_combined_scene.png")
imageio.imwrite(out_png, img)
print(f"Wrote combined-scene render: {out_png}", flush=True)
