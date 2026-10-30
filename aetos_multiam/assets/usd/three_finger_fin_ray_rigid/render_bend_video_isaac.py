"""
Render a video of the finger_1_rigid PhysX articulation (built by
build_finger_1_rigid.py) actually bending under a lateral tip load, inside
Isaac Sim -- the same kind of demo as the MuJoCo version
(three_finger_fin_ray_rigid/render_bend_video.py), but using the PhysX
articulation instead, to directly show the deformation in Isaac Sim rather
than MuJoCo.

Two things had to be found by trial before a frame would render as
anything but a flat gray image:

1. `SimulationApp({"headless": True})` alone (no `experience=` kwarg) still
   left `rep.create.render_product()` + an attached `rgb` annotator
   returning a perfectly uniform (238,238,238) buffer, frame after frame,
   with no error -- not an exposure/lighting problem, just nothing being
   drawn. Switching the capture path to Isaac Sim's own
   `omni.kit.viewport.utility.get_active_viewport()` +
   `capture_viewport_to_file()`, driven by `set_camera_view()` from
   `omni.isaac.core.utils.viewports` (the same viewport machinery the Isaac
   Sim GUI itself uses), produced real images immediately. The replicator
   render-product path may well be fixable with more specific setup, but
   this one is simpler and is known-correct Isaac Sim API usage.
2. The force magnitude/direction found to give a `+2.18deg` tip delta in
   the earlier (non-video) validation (DERIVATION.md section 7) used a
   one-off force application; recalibrating fresh from gravity-rest here
   (see the commented-out calibration sweep this file is based on) found
   -Y at the tip gives a much more dramatic, monotonic curl across all 3
   joints -- e.g. ~14/20/5deg (base/mid/tip) at 0.5N -- which is what this
   video actually uses. +Y at the tip (the sign used in the earlier
   non-video validation run) tends to *straighten* the finger instead
   (consistent with that run's negative base/mid deltas). Neither sign is
   "wrong" -- they're pushing the tip in opposite directions -- but -Y is
   the one that visibly curls the finger further, which is what a video
   demonstrating "flexible deformation" should show.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from isaacsim import SimulationApp

_app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
simulation_app = SimulationApp({"headless": True, "anti_aliasing": 1}, experience=_app_experience)

import omni.usd
from pxr import UsdLux, Gf
from omni.isaac.core.world import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.isaac.dynamic_control import _dynamic_control as dc

from build_finger_1_rigid import build_finger_1_rigid

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
root = build_finger_1_rigid(stage, "/World/finger_1_rigid")

light = UsdLux.DomeLight.Define(stage, "/World/dome_light")
light.CreateIntensityAttr(1200.0)

world.reset()
art = Articulation(prim_path=root)
art.initialize()
rp_tip = RigidPrim(prim_path=f"{root}/finger_1_flex_2")
dc_iface = dc.acquire_dynamic_control_interface()
tip_rb = dc_iface.get_rigid_body(f"{root}/finger_1_flex_2")

# Framed close around the finger (it spans roughly z=0.09..0.19, |y|<0.1).
set_camera_view(eye=[0.14, -0.22, 0.20], target=[0.0, -0.06, 0.15])

FRAMES_DIR = "/tmp/claude-1000/-home-benhua-OmniDrones/a10009f1-db89-4ca5-9a52-909f35b1a9c2/scratchpad/isaac_bend_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)
OUT_MP4 = os.path.join(
    HERE, "..", "..", "meshes", "usd", "three_finger_fin_ray_rigid",
    "finger_1_bend_demo_isaacsim.mp4",
)
os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)

FPS = 30
PHYSICS_HZ = 60
STEPS_PER_FRAME = PHYSICS_HZ // FPS  # 2

vp = get_active_viewport()


def ramp(t, t0, t1, v0, v1):
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    a = (t - t0) / (t1 - t0)
    return v0 + a * (v1 - v0)


TOTAL_T = 7.5
frame_idx = 0
t = 0.0
log = []

while t < TOTAL_T:
    force_mag = 0.0
    if 1.5 <= t < 3.5:
        force_mag = ramp(t, 1.5, 3.5, 0.0, 0.5)
    elif 3.5 <= t < 4.5:
        force_mag = 0.5
    elif 4.5 <= t < 6.5:
        force_mag = ramp(t, 4.5, 6.5, 0.5, 0.0)

    for _ in range(STEPS_PER_FRAME):
        if force_mag > 0.0:
            tip_pos, _ = rp_tip.get_world_pose()
            dc_iface.apply_body_force(tip_rb, (0.0, -force_mag, 0.0), tuple(tip_pos.tolist()), True)
        world.step(render=True)

    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.png")
    capture_viewport_to_file(vp, frame_path)
    simulation_app.update()  # let the async capture flush to disk

    if frame_idx % 15 == 0:
        pos = np.degrees(art.get_joint_positions())
        log.append((round(t, 2), pos.copy(), round(force_mag, 3)))

    frame_idx += 1
    t += 1.0 / FPS

print(f"Rendered {frame_idx} frames to {FRAMES_DIR}", flush=True)
for row in log:
    print(row, flush=True)

# Mux BEFORE simulation_app.close() -- close() appears to terminate the
# process immediately (os._exit()-like), so anything after it never runs;
# confirmed by this exact ordering silently dropping the mux step the
# first time this was written.
import subprocess
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([
    ffmpeg_exe, "-y", "-framerate", str(FPS),
    "-i", os.path.join(FRAMES_DIR, "frame_%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    OUT_MP4,
], check=True)
print("Wrote", OUT_MP4, flush=True)

simulation_app.close()
