"""650 mm Iris + SO-ARM100 interactive hover; macOS: use mjpython."""
from pathlib import Path
import threading,time,json
import mujoco
import mujoco.viewer
from flight_control import HoverController

HERE=Path(__file__).resolve().parent
model=mujoco.MjModel.from_xml_path(str(HERE/'scene.xml'))
data=mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,model.key('hover_home').id)
controller=HoverController(model)
state={'paused':False,'reset':False,'close':False}
lock=threading.Lock()

def key_callback(key):
    with lock:
        if key==32:state['paused']=not state['paused']
        elif key in (82,114):state['reset']=True
        elif key in (71,103):state['close']=not state['close']

print('650 mm Iris + full-size SO-ARM100. Space: pause, R: reset, G: jaw open/close.',flush=True)
print('First five Control sliders: arm angle targets (rad). Rotor thrusts controlled automatically.',flush=True)
with mujoco.viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
    viewer.cam.lookat[:]=[0,0,1.3];viewer.cam.distance=1.8;viewer.cam.azimuth=135;viewer.cam.elevation=-18
    last_status=-1
    while viewer.is_running():
        start=time.perf_counter()
        with lock:
            if state['reset']:
                mujoco.mj_resetDataKeyframe(model,data,model.key('hover_home').id);state['reset']=False
            paused=state['paused'];close=state['close']
        if not paused:
            with viewer.lock():
                data.ctrl[model.actuator('Jaw').id]=-.174 if close else .8
                for _ in range(2):
                    controller.update(data);mujoco.mj_step(model,data,nstep=10)
        viewer.sync()
        if int(data.time)!=last_status:
            last_status=int(data.time)
            (HERE/'viewer_status.json').write_text(json.dumps({'simulation_time':float(data.time),
                'base_position':data.xpos[controller.body_id].tolist(),'warnings':data.warning.number.tolist()}))
        remaining=.02-(time.perf_counter()-start)
        if remaining>0:time.sleep(remaining)

