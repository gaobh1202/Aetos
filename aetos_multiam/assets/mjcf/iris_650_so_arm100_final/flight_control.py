"""Basic joint-aware hover demo; rotor controls are thrusts in newtons."""
from pathlib import Path
import json
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent

class HoverController:
    def __init__(self, model):
        self.model = model
        self.params = json.loads((HERE/'flight_parameters.json').read_text())
        self.A = np.asarray(self.params['allocation_matrix'])
        self.Ainv = np.linalg.inv(self.A)
        self.limits = np.asarray(self.params['max_thrust_N'])
        self.target = np.asarray(self.params['target_base_position'], dtype=float)
        self.mass = self.params['total_mass_kg']
        self.rotor_ids = [model.actuator(f'rotor_{i}_thrust').id for i in range(4)]
        self.body_id = model.body(self.params['model_body']).id
        self.mass_matrix = np.empty((model.nv,model.nv))
        self.last_requested = np.zeros(4)

    def update(self, data):
        mujoco.mj_forward(self.model,data)
        R = data.xmat[self.body_id].reshape(3,3)
        error = self.target - data.xpos[self.body_id]
        acceleration = np.array([3.,3.,6.])*error - np.array([3.,3.,4.])*data.qvel[:3]
        acceleration[:2] = np.clip(acceleration[:2],-1.5,1.5)
        acceleration[2] = np.clip(acceleration[2],-3,3)
        desired_force = self.mass * (acceleration + np.array([0,0,9.81]))
        z = desired_force/np.linalg.norm(desired_force)
        y = np.cross(z,np.array([1.,0.,0.]));y /= np.linalg.norm(y)
        desired_R = np.column_stack((np.cross(y,z),y,z))
        skew = .5*(desired_R.T@R - R.T@desired_R)
        attitude_error = np.array([skew[2,1],skew[0,2],skew[1,0]])
        # Free-joint rotational velocities are expressed in the body frame.
        alpha = -20*attitude_error - 9*data.qvel[3:6]
        mujoco.mj_fullM(self.model,data,self.mass_matrix)
        torque = self.mass_matrix[3:6,3:6]@alpha + data.qfrc_bias[3:6]
        collective = max(float(desired_force@R[:,2]),0.)
        self.last_requested = self.Ainv @ np.r_[collective,torque]
        data.ctrl[self.rotor_ids] = np.clip(self.last_requested,0,self.limits)

