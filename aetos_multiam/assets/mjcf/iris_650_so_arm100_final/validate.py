from pathlib import Path
import json
import numpy as np
import mujoco
from flight_control import HoverController

HERE=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path(str(HERE/'scene.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,m.key('hover_home').id);mujoco.mj_forward(m,d)
c=HoverController(m)
assert m.nq==13 and m.nv==12 and m.nu==10
positions=np.array([d.body(f'rotor_{i}').xpos for i in range(4)])
actual=float(np.linalg.norm(positions[:,None]-positions[None,:],axis=-1).max())
assert abs(actual-.650)<1e-8
assert np.linalg.norm(d.site('arm_mount').xpos-d.site('so_arm100_mount').xpos)<1e-9
tcp_in_body=d.xmat[c.body_id].reshape(3,3).T@(d.site('grasp_tcp').xpos-d.site('arm_mount').xpos)
assert tcp_in_body[0]>.15 and abs(tcp_in_body[1])<.01, tcp_in_body
assert d.ncon==0,[(m.geom(x.geom1).name,m.geom(x.geom2).name,x.dist) for x in d.contact]
home=np.array(c.params['home_angles_rad'])
stats=[];max_error=0.;max_tilt=0.;saturated=0
for name,steps in [('stationary',200),('arm_and_jaw_motion',400),('settle',400)]:
    for i in range(steps):
        if name=='arm_and_jaw_motion':
            t=(i+1)/steps;s=t*t*(3-2*t)
            target=home+np.array([.2,.1,.15,0,.1,0])*s
            target[-1]=.8*(1-s)-.174*s
            d.ctrl[:6]=target
        c.update(d)
        saturated+=int(np.any((c.last_requested<0)|(c.last_requested>c.limits)))
        mujoco.mj_step(m,d,nstep=10)
        assert np.isfinite(d.qpos).all() and not d.warning.number.any()
        e=float(np.linalg.norm(d.xpos[c.body_id]-c.target))
        tilt=float(np.arccos(np.clip(d.xmat[c.body_id].reshape(3,3)[2,2],-1,1)))
        max_error=max(max_error,e);max_tilt=max(max_tilt,tilt)
    row={'segment':name,'time_s':float(d.time),'base_position_m':d.xpos[c.body_id].tolist(),
         'position_error_m':e,'tilt_deg':float(np.rad2deg(tilt)),'contacts':d.ncon,
         'thrusts_N':d.ctrl[c.rotor_ids].tolist(),'joint_angles_rad':d.qpos[7:].tolist()}
    stats.append(row);print(json.dumps(row),flush=True)
assert max_error<.1 and max_tilt<np.deg2rad(10)
report={'actual_max_wheelbase_m':actual,'total_mass_kg':c.mass,'home_tcp_from_mount_body_m':tcp_in_body.tolist(),'test_duration_s':float(d.time),
        'max_position_error_m':max_error,'max_tilt_deg':float(np.rad2deg(max_tilt)),
        'saturated_updates':saturated,'runtime_warning_counts':d.warning.number.tolist(),'history':stats}
(HERE/'validation.json').write_text(json.dumps(report,indent=2))
print('PASS',json.dumps({k:v for k,v in report.items()if k!='history'}))
