"""In-place fixups for the already-built iris_650.usd (see import_drone_usd.py).

Two things the as-shipped file got wrong, both of which only bite once the
drone is part of a *merged* articulation (i.e. with an arm mounted) or once
real thrust is commanded:

1. ``ArticulationRootAPI`` sat on the ``/Iris650`` wrapper Xform instead of
   on ``base_link``. PhysX then picks the root LINK itself, and for
   ``iris650_soarm100`` it picked the mounted arm's own first body
   (``upper_arm`` -- body index 0 in the merged articulation). Isaac's
   post-reset then writes the default root pose onto THAT body, which
   rigidly rotated the whole drone+arm assembly by the arm's own mount
   frame: measured base_link world quat (0.5,-0.5,0.5,-0.5) at spawn, i.e.
   the drone standing on its edge, rotors spread +-0.29m VERTICALLY. Every
   earlier "axis misalignment" seen on this asset was that rotation, not a
   convention problem -- the USD's own authored frame is textbook z-up
   (verified: rotors at (+-0.171, +-0.2895/+-0.2632, +0.0303) relative to
   base_link, matching iris_650.yaml's rotor_angles/arm_lengths exactly).
   Fixed by moving the API onto base_link, matching amov_drone's layout
   (and what import_drone_usd.py itself already intends).

2. Every body carried the MJCF's own +1.5m z placement, so spawning at
   (0,0,1) actually put the drone at z=2.5. Subtract base_link's authored
   translation from all five bodies (pure translation; every rotation is
   already identity and all joint frames are body-relative, so relative
   geometry is untouched).

Idempotent -- safe to re-run.
"""
from pathlib import Path

from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

DEST = Path(__file__).resolve().parents[2] / "meshes" / "usd" / "iris_650.usd"
BODIES = ("base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3")

stage = Usd.Stage.Open(str(DEST))

wrapper = stage.GetPrimAtPath("/Iris650")
base = stage.GetPrimAtPath("/Iris650/base_link")
if wrapper.HasAPI(UsdPhysics.ArticulationRootAPI):
    wrapper.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print("removed ArticulationRootAPI from /Iris650", flush=True)
if not base.HasAPI(UsdPhysics.ArticulationRootAPI):
    UsdPhysics.ArticulationRootAPI.Apply(base)
    print("applied ArticulationRootAPI to /Iris650/base_link", flush=True)


def _transform_op(prim):
    op, = [o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()
           if o.GetOpName() == "xformOp:transform"]
    return op


offset = _transform_op(base).Get().ExtractTranslation()
if Gf.Vec3d(offset).GetLength() > 1e-9:
    for name in BODIES:
        op = _transform_op(stage.GetPrimAtPath(f"/Iris650/{name}"))
        m = op.Get()
        m.SetTranslateOnly(m.ExtractTranslation() - offset)
        op.Set(m)
        print(f"  {name} -> {m.ExtractTranslation()}", flush=True)
    print(f"removed baked body offset {offset}", flush=True)

stage.GetRootLayer().Save()
print("saved", DEST, flush=True)
app.close()
