"""Run with macOS mjpython. Space pauses; R resets; G toggles the gripper."""
from pathlib import Path
import threading
import time
import mujoco
import mujoco.viewer
from flight_control import HoverController

HERE = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(HERE/'scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,model.key('hover_open').id)
controller = HoverController(model)
state = {'paused':False,'reset':False,'close':False}
state_lock = threading.Lock()

def on_key(key):
    with state_lock:
        if key == 32:state['paused'] = not state['paused']
        elif key in (82,114):state['reset'] = True
        elif key in (71,103):state['close'] = not state['close']

print('Hummingbird 340 mm + Dynamixel 2R + three-finger gripper',flush=True)
print('Hover controller enabled. Space: pause; R: reset; G: open/close gripper.',flush=True)
print('Arm R1/R2 targets can be changed in the Control panel.',flush=True)
with mujoco.viewer.launch_passive(model,data,key_callback=on_key) as viewer:
    viewer.cam.lookat[:] = [0,0,1.0]
    viewer.cam.distance = 1.15
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -12
    status = HERE/'viewer_status.json'
    last_status = -1
    while viewer.is_running():
        start = time.perf_counter()
        with state_lock:
            if state['reset']:
                mujoco.mj_resetDataKeyframe(model,data,model.key('hover_open').id)
                state['reset']=False
            paused = state['paused']
            close = state['close']
        if not paused:
            data.ctrl[model.actuator('gripper_close').id] = .6981317008 if close else 0.
            with viewer.lock():
                # 100 Hz flight feedback; display at 50 Hz.
                for _ in range(2):
                    controller.update(data)
                    mujoco.mj_step(model,data,nstep=200)
        viewer.sync()
        if int(data.time) != last_status:
            import json
            last_status=int(data.time)
            status.write_text(json.dumps({'simulation_time':float(data.time),
                'drone_position':data.xpos[controller.body_id].tolist(),
                'runtime_warnings':data.warning.number.tolist()}))
        # Limit real-time rate; slow hardware may run below real time.
        remaining=.02-(time.perf_counter()-start)
        if remaining>0:time.sleep(remaining)

