# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""HoverShelfGrasp: Hover task whose scene additionally carries the
raids-am-simulator shelf_grasp furniture/props, so a trained hover policy can
fly the (separately spawned) AmovDrone into that scene and hold position
there. See scripts/run_shelf_grasp.py for the fly-in -> arm-replay rollout
logic that drives this task at runtime.
"""

import sys
from pathlib import Path

import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .hover import Hover

# raids-am-simulator/raids_am_Scene/build_isaacsim_scene.py owns the shelf/prop
# USD paths and the hand-tuned poses/scales for this scene -- imported (not
# duplicated) so the two scenes can't drift apart. That module only defines
# constants/functions at import time (its scene-build side effects live in
# main(), guarded by `if __name__ == "__main__"`), so importing it here is safe.
_RAIDS_SCENE_DIR = Path("/home/benhua/raids-am-simulator/raids_am_Scene")
if str(_RAIDS_SCENE_DIR) not in sys.path:
    sys.path.insert(0, str(_RAIDS_SCENE_DIR))
import build_isaacsim_scene as shelf_scene  # noqa: E402

# Prim names under {env_root}/Objects/ -- matches raids_am_Scene/replay/
# replay_isaacsim.py's OBJECT_PRIM_ROOTS naming so scripts/run_shelf_grasp.py
# can resolve the same rigid bodies for per-frame teleporting.
OBJECT_PRIM_NAMES = {
    "gearwheel": "Gearwheel",
    "button": "Button",
    "power_drill": "PowerDrill",
    "tote_box": "ToteBox",
}


def _strip_articulation_roots(stage, root_path):
    """Some prop assets carry their own ArticulationRootAPI from an internal
    mechanism (e.g. Button's E_body_16 roots the press-plunger's prismatic
    joints). These objects are only ever kinematically teleported wholesale
    here -- their internal joints are never driven -- and PhysX's GPU tensor
    pipeline (which IsaacEnv uses, unlike the plain omni.isaac.core.World path
    replay_isaacsim.py uses) refuses to create a rigid-body view for a
    kinematic prim that still carries ArticulationRootAPI: it logs
    "articulation root will be ignored" and then RigidPrim.initialize() later
    fails outright because no matching body exists in the tensor view. Strip
    the API before flipping the prim kinematic so PhysX treats it as a plain
    kinematic rigid body."""
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _add_reference_abs(stage, path, usd_path, pos, yaw_deg=0.0, scale=1.0):
    """Same as build_isaacsim_scene.add_reference(), but references the asset
    by absolute path instead of a path relative to that script's own file --
    this scene lives on an in-memory runtime stage, not a saved .usd next to
    the asset files, so a relative reference wouldn't resolve."""
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(usd_path))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if yaw_deg:
        xform.AddRotateZOp().Set(yaw_deg)
    if scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    return prim


class HoverShelfGrasp(Hover):
    def _design_scene(self):
        global_prim_paths = super()._design_scene()

        stage = prim_utils.get_current_stage()
        env_root = self.template_env_ns

        shelf_path = f"{env_root}/Shelf"
        _add_reference_abs(stage, shelf_path, shelf_scene.SHELF_USD, shelf_scene.SHELF_POS)
        shelf_scene.normalize_rigid_root_xform(
            stage, f"{shelf_path}/E_RootNode001_2", scale=shelf_scene.SHELF_SCALE
        )
        _strip_articulation_roots(stage, shelf_path)
        shelf_scene.make_kinematic(stage, shelf_path)

        objects_root = f"{env_root}/Objects"

        gearwheel_path = f"{objects_root}/{OBJECT_PRIM_NAMES['gearwheel']}"
        _add_reference_abs(
            stage, gearwheel_path, shelf_scene.GEARWHEEL_USD,
            shelf_scene.GEARWHEEL_POS, scale=shelf_scene.GEARWHEEL_SCALE,
        )
        shelf_scene.normalize_rigid_root_xform(stage, f"{gearwheel_path}/E_body_24")

        button_path = f"{objects_root}/{OBJECT_PRIM_NAMES['button']}"
        _add_reference_abs(
            stage, button_path, shelf_scene.BUTTON_USD,
            shelf_scene.BUTTON_POS, scale=shelf_scene.BUTTON_SCALE,
        )
        shelf_scene.normalize_rigid_root_xform(stage, f"{button_path}/E_body_16")
        shelf_scene.normalize_rigid_root_xform(stage, f"{button_path}/E_button_01_13")
        shelf_scene.normalize_rigid_root_xform(stage, f"{button_path}/E_button_02_6")

        drill_path = f"{objects_root}/{OBJECT_PRIM_NAMES['power_drill']}"
        _add_reference_abs(
            stage, drill_path, shelf_scene.DRILL_USD,
            shelf_scene.DRILL_POS, yaw_deg=shelf_scene.DRILL_YAW_DEG,
        )
        shelf_scene.deactivate_prim(stage, f"{drill_path}/power_drill/FixedJoint")

        tote_path = f"{objects_root}/{OBJECT_PRIM_NAMES['tote_box']}"
        _add_reference_abs(
            stage, tote_path, shelf_scene.TOTE_BOX_USD, shelf_scene.TOTE_BOX_POS,
            yaw_deg=shelf_scene.TOTE_BOX_YAW_DEG, scale=shelf_scene.TOTE_BOX_SCALE,
        )
        shelf_scene.normalize_rigid_root_xform(stage, f"{tote_path}/E_tote_box_1")

        for prim_name in OBJECT_PRIM_NAMES.values():
            _strip_articulation_roots(stage, f"{objects_root}/{prim_name}")
            shelf_scene.make_kinematic(stage, f"{objects_root}/{prim_name}")

        return global_prim_paths
