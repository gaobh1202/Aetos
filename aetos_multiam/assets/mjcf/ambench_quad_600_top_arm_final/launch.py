"""Kinematic MuJoCo inspection: Control sliders move the original arm joints."""
from pathlib import Path
import json
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(HERE / 'scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key('initial').id)
joint_actuators = [(i, int(model.actuator_trnid[i, 0])) for i in range(model.nu)
                   if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT]
state = {'reset': False, 'grip': None}
lock = threading.Lock()


def on_key(key: int) -> None:
    with lock:
        if key in (82, 114):
            state['reset'] = True
        elif key in (71, 103):
            state['grip'] = not state['grip']


print('MuJoCo robot viewer: original arm and gripper geometry.', flush=True)
print('Kinematic display, fixed vehicle base, gravity disabled. Control sliders: joint targets.', flush=True)
print('R: reset. G: open/close grippers. Mouse: orbit/zoom.', flush=True)
with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    viewer.cam.lookat[:] = model.stat.center
    viewer.cam.distance = 2.7 if model.nq > 6 else 1.8
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    frame = 0
    while viewer.is_running():
        start = time.perf_counter()
        with lock:
            if state['reset']:
                mujoco.mj_resetDataKeyframe(model, data, model.key('initial').id)
                state['reset'] = False
                state['grip'] = None
            gripper = state['grip']
        with viewer.lock():
            for actuator, joint in joint_actuators:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                low, high = model.jnt_range[joint]
                if gripper is not None and 'finger_joint' in name:
                    data.ctrl[actuator] = min(.03, high) if gripper else max(.001, low)
                data.qpos[model.jnt_qposadr[joint]] = np.clip(data.ctrl[actuator], low, high)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
        viewer.sync()
        frame += 1
        if frame % 50 == 1:
            (HERE / 'viewer_status.json').write_text(json.dumps({'mode': 'kinematic', 'frames': frame, 'joints': model.njnt, 'finite': bool(np.isfinite(data.qpos).all())}))
        remaining = .02 - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
