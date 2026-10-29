"""
Render a short video of the rigid Fin-Ray finger_1 approximation actually
bending, to visually confirm the passive compliance (section 4 of
DERIVATION.md) looks like real Fin-Ray behavior: the finger closes toward
a pose, presses into a load, the distal (softest) segments curl the most
and conform around it, then it springs back when the load is released.

Timeline (simulated, real physics -- not keyframed/faked):
  t in [0.0, 1.0): settle under gravity only, actuator at rest (0 deg).
  t in [1.0, 2.0): actuator ramps 0 -> 0.35 rad (~20deg), closing the finger.
  t in [2.0, 2.5): hold the actuator; nothing pressing on it yet (baseline close pose).
  t in [2.5, 4.0): a lateral force at the tip ramps 0 -> 0.6N, simulating
                   contact with an object -- the distal segments should
                   curl noticeably more than the base segment (section 4's
                   stiffness ratio: tip is ~22x softer than the base).
  t in [4.0, 5.0): hold the load at 0.6N.
  t in [5.0, 6.5): both the force and the actuator ramp back to 0 -- spring-back.
  t in [6.5, 7.5): settle at rest again.

Renders at 30fps via MuJoCo's offscreen renderer (already confirmed working
in this environment) and muxes to mp4 with ffmpeg.
"""
import os
import subprocess
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = "/tmp/claude-1000/-home-benhua-OmniDrones/a10009f1-db89-4ca5-9a52-909f35b1a9c2/scratchpad/finger_bend_frames"
OUT_MP4 = os.path.join(HERE, "finger_1_bend_demo.mp4")

os.makedirs(FRAMES_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=960)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

tip_site = model.site("finger_1_tip_rigid").id
tip_body = model.body("finger_1_flex_2").id
hinge_act = model.actuator("finger_1_close").id

FPS = 30
DT_FRAME = 1.0 / FPS
TOTAL_T = 7.5


def ramp(t, t0, t1, v0, v1):
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    a = (t - t0) / (t1 - t0)
    return v0 + a * (v1 - v0)


frame_idx = 0
t = 0.0
joint_angles_log = []

while t < TOTAL_T:
    actuator_target = ramp(t, 1.0, 2.0, 0.0, 0.35)
    data.ctrl[hinge_act] = actuator_target
    if t >= 5.0:
        actuator_target_release = ramp(t, 5.0, 6.5, 0.35, 0.0)
        data.ctrl[hinge_act] = actuator_target_release

    force_mag = 0.0
    if 2.5 <= t < 4.0:
        force_mag = ramp(t, 2.5, 4.0, 0.0, 0.15)
    elif 4.0 <= t < 5.0:
        force_mag = 0.15
    elif 5.0 <= t < 6.5:
        force_mag = ramp(t, 5.0, 6.5, 0.15, 0.0)

    while data.time < t + DT_FRAME:
        data.qfrc_applied[:] = 0.0
        if force_mag > 0.0:
            tip_pos = data.site_xpos[tip_site].copy()
            # +Y here (not -Y): these joints use axis="-1 0 0" in the MJCF
            # (opposite sign convention from the Isaac Sim rebuild's
            # literal axis="X"), so the sign that produces positive,
            # within-limit curling is flipped relative to that script.
            force = np.array([0.0, force_mag, 0.0])
            mujoco.mj_applyFT(model, data, force, np.zeros(3), tip_pos, tip_body, data.qfrc_applied)
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="overview")
    img = renderer.render()
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.png"), img)

    joint_angles_log.append((round(t, 2), np.degrees(data.qpos[-3:]).copy(), round(actuator_target, 3), round(force_mag, 3)))
    frame_idx += 1
    t += DT_FRAME

print(f"Rendered {frame_idx} frames to {FRAMES_DIR}")
for row in joint_angles_log[::15]:
    print(row)

import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([
    ffmpeg_exe, "-y", "-framerate", str(FPS),
    "-i", os.path.join(FRAMES_DIR, "frame_%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    OUT_MP4,
], check=True)
print("Wrote", OUT_MP4)
