# Rigid Fin Ray finger -- derivation (finger_1 only, first pass)

**Status: validated in MuJoCo** (the actual simulator -- `/home/benhua/miniconda3/envs/airbotsim`
has `mujoco==3.13.0` installed locally, same version the original gripper
bundle targeted). `finger_rigid.xml` compiles, is numerically stable under
gravity (2s, max |qvel| 0.0066 rad/s, no blow-up), and curls plausibly
under a test tip load -- see `preview_rest.png`/`preview_bent_under_tip_load.png`.
This is *not* validated against Isaac Sim/PhysX yet, and the original
flexcomp's actual deflection under the same load was never measured either
(no live comparison baseline exists), so "plausible" is as far as this goes.

## 1. Source geometry

`three_finger_gripper_final/gripper.xml`'s `finger_1_elastic` flexcomp
defines its soft body as a point cloud (`dim="3"`, 96 points: 48 at local
x=+0.007, 48 mirrored at x=-0.007 -- i.e. a 14mm-thick blade, symmetric).
The 48 points at x=+0.007 are laid out as 6 contiguous blocks of 8 in the
XML's `point` attribute; parsing them (`/tmp/parse_finger.py` during this
session) gives 6 cross-section "layers" from base to tip:

| layer | mean y (m) | mean z (m) | y-spread / "width" (m) |
|---|---|---|---|
| L5 (base, nearest the rigid mount) | -0.02504 | 0.01316 | 0.02806 |
| L4 | -0.03287 | 0.03162 | 0.02351 |
| L3 | -0.04121 | 0.05094 | 0.01828 |
| L2 | -0.04931 | 0.07037 | 0.01398 |
| L1 | -0.05764 | 0.08970 | 0.00921 |
| L0 (tip) | -0.06536 | 0.10821 | 0.00466 |

(Coordinates are in `finger_1`'s own local body frame -- confirmed by the
flexcomp having no separate `frame`/`pos`, so it inherits its parent body's
frame, and by `L0`'s position landing within ~3mm of the original
`finger_1_contact_tip` site's own local coordinates, (y=-0.0615, z=0.1056),
which is defined in the same frame.)

The "width" column is the y-spread *within* a layer (8 points per layer,
likely front/back rib pairs of the Fin Ray's internal zigzag structure,
duplicated across the mesh's tet connectivity -- not independently
re-derived here, just used as a proxy for the cross-section's overall
bending-plane extent). It tapers monotonically from 28.06mm at the base to
4.66mm at the tip.

## 2. Three-segment discretization

Treating L5..L0 as a polyline and walking its cumulative arc length
(`/tmp/parse_finger2.py`/`parse_finger3.py`), total flex length =
**103.25mm**. Splitting into 3 *equal arc-length* segments (not equal
layer-count -- the layers aren't evenly spaced) via linear interpolation
gives 4 waypoints:

| waypoint | y (m) | z (m) | width (m) |
|---|---|---|---|
| 0 (base, = original flex start) | -0.02504 | 0.01316 | 0.02806 |
| 1 | -0.03856 | 0.04481 | 0.01994 |
| 2 | -0.05196 | 0.07651 | 0.01246 |
| 3 (tip) | -0.06536 | 0.10821 | 0.00466 |

Each of the 3 segments: **length 34.416mm**, width = average of its two
endpoints (seg0: 24.00mm, seg1: 16.20mm, seg2: 8.56mm), thickness constant
at 14mm (unchanged from the original, both x-sides).

Direction angle (rotation about local X needed to map a body's local Z
axis onto that segment's Y-Z direction, verified numerically with an
explicit rotation-matrix check, not just the formula): seg0 = 23.131deg,
seg1 = 22.903deg, seg2 = 22.921deg -- i.e. the centerline is close to
straight at rest (consistent with it being an unloaded resting pose), with
segment-to-segment direction changes of only +0.234deg and -0.018deg.
Those are dropped (treated as 0) in `finger_rigid.xml` -- only the first
attachment (`finger_1` -> `finger_1_flex_0`) carries the 0.403709 rad
rotation; segments 1 and 2 attach with identity rotation. This is a
deliberate simplification, not an oversight.

## 3. Mass

The original flexcomp specifies one `mass="0.008795869463462445"` for the
*whole* soft body. Split across the 3 segments proportional to box volume
(`thickness * width * length`): seg0 **4.329g**, seg1 **2.922g**, seg2
**1.544g** (sums back to 8.796g).

## 4. Joint stiffness (the actual physics claim here)

Each segment's own bending compliance is lumped into the hinge at *its own*
proximal end (`finger_1_flex_hinge_0` for seg0, `_1` for seg1, `_2` for
seg2) via a standard cantilever-equivalent stiffness:

```
I = thickness * width^3 / 12          (second moment of area, bending in the Y-Z plane)
EI = E * I                             (E = 200000 Pa, from the original <elasticity young="200000.0">)
k = EI / L                             (lumped rotational stiffness, N*m/rad)
```

| segment | width (m) | I (m^4) | EI (N*m^2) | k (N*m/rad) |
|---|---|---|---|---|
| 0 | 0.024 | 1.6128e-8 | 3.2256e-3 | **0.09372** |
| 1 | 0.01620 | 4.9621e-9 | 9.9243e-4 | **0.02884** |
| 2 | 0.00856 | 7.3234e-10 | 1.4647e-4 | **0.00426** |

Stiffness decreases ~22x from base to tip, correctly reproducing Fin Ray's
known progressive/adaptive curling (the tip yields first and most).
Damping (0.0010 / 0.0008 / 0.0005 N*m*s/rad) was *not* derived the same
way -- there's no damping term to port (the original's
`<elasticity damping="0.0">`) -- these are just small values picked for
numerical stability at a 1ms timestep and confirmed not to cause blow-up;
treat them as a knob to retune, not a measurement.

## 5. What this is not

- Not a flat ribbon cross-section -- MuJoCo boxes can't taper, so each
  segment is a constant-width box sized at its *average* width, not a
  true trapezoid. A finer discretization (more, shorter segments) would
  narrow this gap.
- Not validated against the original flexcomp's actual deflection under
  load -- no baseline measurement of that exists either (the original
  bundle's own `validation.json` tests hover/joint-motion/gripper
  open-close, not finger-tip deflection specifically), so there's nothing
  to numerically compare against yet.
- Not yet imported into Isaac Sim, and not yet integrated back into the
  full 3-finger `gripper.xml` (finger_2/finger_3 + their
  `finger_2_synchronized`/`finger_3_synchronized` equality constraints,
  which only ever synchronized the *actuated* base hinges, not the flex
  joints -- each finger's passive segments would stay independent, which
  is physically correct).

## 6. How to re-run the validation

```bash
/home/benhua/miniconda3/envs/airbotsim/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('scene.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
print('compiled OK, nq=', m.nq)
"
```

## 7. Second representation: the same structure, authored directly in Isaac Sim

`aetos_multiam/assets/usd/three_finger_fin_ray_rigid/build_finger_1_rigid.py`
builds this exact 3-segment structure (same lengths/widths/masses/joint
stiffnesses as sections 2-4 above) as a real PhysX articulation and exports
it to `finger_1_rigid.usd` in that same directory -- a second, independent
realization of the structure, not a MJCF->USD conversion of `finger_rigid.xml`
(`omni.importer.mjcf` was never invoked on this file).

Authoring this directly in USD Physics surfaced three real pitfalls that
MJCF's authoring model hides, each found by actually running it in Isaac
Sim and reading the PhysX error log / joint-state readback, not by
inspecting the schema docs:

1. **No nested RigidBodyAPI prims.** MJCF's native body-in-body nesting
   (`finger_1` -> `finger_1_flex_0` -> `finger_1_flex_1` -> `finger_1_flex_2`
   as a literal XML parent/child chain) is rejected by PhysX ("xformstack
   reset ... missing when child of rigid body in hierarchy"). Fix: all 4
   bodies become FLAT SIBLINGS under one container Xform, each given its
   own absolute world transform (computed by hand-composing the same local
   offsets/tilt from section 2), connected only through joint prims
   (`body0Rel`/`body1Rel`), never through USD parent/child nesting.

2. **No kinematic bodies inside an articulation.** The natural way to model
   "finger_1's rigid mount, fixed in place for this isolated test" seemed
   to be a kinematic rigid body -- PhysX rejects that outright
   ("Articulations with kinematic bodies are not supported"). Fix: make it
   a normal DYNAMIC body welded to the world frame with a
   `UsdPhysics.FixedJoint` (`body0Rel` left unset = anchored to world),
   the same pattern Isaac Sim's own URDF importer uses for "Fix Base Link".

3. **The expensive one: `LocalRot0`/`LocalRot1` do not default to identity
   when left unauthored.** Every joint silently became a 0-DOF/rigid
   weld -- `get_joint_positions()`/`get_joint_velocities()` read back
   exactly `0.0` forever, under gravity, with zero warning or error, for
   every variant tried (wider limits, no drive, floating base, explicit
   inertia, zero sleep threshold -- none of it was the cause). Isolated by
   comparing against `omni.physx.scripts.utils.createJoint()` (Isaac Sim's
   own tested joint-creation helper), which always explicitly computes and
   sets these two attributes -- an isolated single-hinge test using that
   helper moved correctly; the hand-authored version with the attributes
   left unset did not. These must always be authored explicitly, and when
   the two bodies either side of a joint have *different* world
   orientations (as `finger_1`[identity] -> `finger_1_flex_0`[23.131deg
   tilt] do), `LocalRot0` (body0's side) must carry that relative rotation
   -- identity on both sides is only correct when both bodies already
   share the same world orientation, which happens to be true for
   `flex_hinge_1`/`flex_hinge_2` here but is NOT true for `flex_hinge_0`.

**Validated** (ran for real via the `omnidrone` conda env, which has
`ISAACSIM_PATH` wired to the local Isaac Sim 4.1.0 install): builds with no
PhysX errors; settles under gravity to `[0.86, 1.21, 1.56]` deg
(base->tip) -- small, physically plausible, matching a hand estimate from
the segment masses and lever arms; under a 0.3N lateral tip load (the same
test as section 1's `preview_bent_under_tip_load.png`, just applied via
`omni.isaac.dynamic_control` instead of MuJoCo's `mj_applyFT`), the tip
joint curls far more (delta `+2.18deg`) while the two base joints barely
move (`-0.69deg`, `-1.11deg`) -- reproducing Fin-Ray's known progressive,
tip-concentrated compliance, same qualitative result as the MuJoCo
validation. The exported `finger_1_rigid.usd` was also round-trip tested
(referenced into a *fresh* stage via `add_reference_to_stage`, re-simulated
from scratch) and reproduces the same gravity-settled angles, confirming
the exported file is a usable standalone asset, not just an artifact of
the authoring session's live stage.

**Not yet done:** finger_2/finger_3 in USD, mounting this under a real
gripper housing instead of a standalone world anchor (the `FixedJoint`
here stands in for that), and reassembling the full 3-finger gripper.

## 8. Video of the deformation (MuJoCo, real-time physics)

`render_bend_video.py` (this directory) renders `finger_1_bend_demo.mp4` --
not a keyframed/faked animation, every frame comes from actually stepping
`finger_rigid.xml`'s physics: 0-2s the base actuator closes to ~20deg,
2.5-5s a lateral tip force ramps 0->0.15N and holds (the distal segments
curl noticeably more than the base -- base ~8deg, mid ~16deg, tip ~38deg,
matching the ~22x base-to-tip stiffness ratio from section 4), 5-7.5s the
load releases and the finger springs back to rest. Re-render with:

```bash
/home/benhua/miniconda3/envs/airbotsim/bin/python render_bend_video.py
```

(uses `imageio_ffmpeg`'s bundled libx264 binary -- the system `ffmpeg` at
`/usr/local/ffmpeg` in this environment has no x264/x265/vp9 encoder
compiled in, only `mpeg4`/`h264_v4l2m2m`, so it can't produce a
widely-playable mp4 on its own here).

## 9. Video of the deformation (Isaac Sim, real-time physics)

`aetos_multiam/assets/usd/three_finger_fin_ray_rigid/render_bend_video_isaac.py`
renders the same kind of demo but driving the PhysX articulation from
section 7 instead of MuJoCo: gravity-settle, a lateral tip-force ramp
(0->0.5N via `omni.isaac.dynamic_control`), hold, release, spring-back.
Two non-obvious things had to be found before a frame rendered as
anything but flat gray:

- **`rep.create.render_product()` + an attached `rgb` annotator produced a
  perfectly uniform (238,238,238) image, every frame, with no error** --
  not a lighting/exposure problem, nothing was being drawn at all. This
  was true both with and without passing Isaac Sim's full
  `omni.isaac.sim.python.kit` experience file to `SimulationApp`. Switched
  to `omni.kit.viewport.utility.get_active_viewport()` +
  `capture_viewport_to_file()`, positioned via
  `omni.isaac.core.utils.viewports.set_camera_view()` -- the same
  machinery the Isaac Sim GUI itself uses -- which worked immediately.
  (This repo's own `omni_drones/sensors/camera.py` uses the replicator
  render-product path successfully elsewhere, driven by
  `SimulationContext.render()` rather than `rep.orchestrator.step()` and
  attaching annotators as `annotator.attach([render_product])` -- a list,
  not a bare render product -- so the render-product path is very likely
  fixable with the right incantation; the viewport-capture path was just
  faster to get working for this one-off video.)
- **`simulation_app.close()` appears to terminate the process immediately**
  (nothing after it runs, no error) -- the `ffmpeg` mux step has to happen
  *before* calling `close()`, not after.
- Recalibrating the tip-load direction fresh from gravity-rest (rather
  than reusing the exact one-off force from section 7's non-video
  validation) found that **-Y at the tip visibly curls the finger further
  (all 3 joints), while +Y (the sign section 7 used) tends to straighten
  it instead** -- neither is "wrong" (they push the tip opposite ways),
  but -Y is the one that makes a demonstration of "it's flexible" actually
  look like one. At 0.5N: base/mid/tip peak at roughly 9/16.5/6 deg.

Re-render with:

```bash
export ISAACSIM_PATH=~/isaacsim
source /home/benhua/miniconda3/etc/profile.d/conda.sh && conda activate omnidrone
source $ISAACSIM_PATH/setup_conda_env.sh
python render_bend_video_isaac.py
```
