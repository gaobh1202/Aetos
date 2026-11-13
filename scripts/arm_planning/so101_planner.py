"""cuRobo planner for the SO101 grasping the carry-box handle (armplan env).

Everything is expressed in the arm-base (URDF ``base_link``) frame; joint angles
in and out use the simulator's convention (see so101_common.py).
"""
import os.path as osp

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from so101_common import Q_URDF_MINUS_SIM, URDF_DIR, T, load_demo, nominal_grasp

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def pose7(m):  # 4x4 -> [x y z, qw qx qy qz]
    return [*m[:3, 3], *np.roll(R.from_matrix(m[:3, :3]).as_quat(), 1)]


class GraspPlanner:
    def __init__(self, free_bar_rotation=True, bar_slide=0.08):
        """bar_slide: how far (m) the grasp point may sit from the bar centre along the bar
        (26 cm long between its posts); 0 pins it to the centre as in the demos."""
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import PoseErrorFrame, ToolPoseCriteria

        robot = yaml.safe_load(open(osp.join(URDF_DIR, "so101_curobo.yml")))
        robot["kinematics"]["asset_root_path"] = URDF_DIR
        robot["kinematics"]["urdf_path"] = osp.join(URDF_DIR, "so101_grasp.urdf")
        # Demo values: arm-base pose, handle frame (bar along y), hand-object pose, arm command (zeros).
        self.armbase_demo, self.handle, self.handle_grasp, self.q_demo = nominal_grasp(load_demo(0), 0)
        self.box = self.handle @ T((0.42, 0.0, -0.34 + 0.11), (1, 0, 0, 0))  # centre of the 0.90 x 0.40 x 0.22 body
        self.p = MotionPlanner(MotionPlannerCfg.create(
            robot=robot, scene_model=self._scene(self.armbase_demo), collision_cache={"cuboid": 4},
            graph_planner_algorithm="rrt"))
        self.p.warmup(enable_graph=True, num_warmup_iterations=5)
        self.bar = self.handle_grasp[:3, :3].T @ [0.0, 1.0, 0.0]  # bar axis in the grasp frame
        if free_bar_rotation:  # the bar is a cylinder: leave the rotation about it to the 5-DoF arm
            bar = torch.tensor(self.bar, dtype=torch.float32)
            proj = (torch.eye(3) - torch.outer(bar, bar)).to(self.p.device_cfg.device)
            self.p.update_tool_pose_criteria({"grasp_frame": ToolPoseCriteria(
                terminal_rotation_projector=proj, path_rotation_projector=proj * 0.0,
                # Sliding along the bar is free for the optimiser; plan() rejects slides past bar_slide.
                terminal_position_projector=proj if bar_slide > 0 else None,
                error_frame=PoseErrorFrame.GOAL, device_cfg=self.p.device_cfg)})
        self.bar_slide = bar_slide if free_bar_rotation else 0.0
        self.names = self.p.joint_names
        self.arm = [self.names.index(j) for j in ARM]

    def _scene(self, armbase):
        from curobo.scene import Cuboid, Scene

        return Scene(cuboid=[
            Cuboid(name="box", dims=[0.90, 0.40, 0.22], pose=pose7(np.linalg.inv(armbase) @ self.box)),
            # The airframe above the mount (arm-base z points down): a slab the arm may not swing up into.
            # shortcut: one box 4 cm clear of the mount plane, not the real frame and rotor discs.
            Cuboid(name="drone", dims=[0.70, 0.70, 0.10], pose=[0.0, 0.0, -0.09, 1, 0, 0, 0]),
        ])

    def _state(self, q_sim):
        from curobo.types import JointState

        q = torch.zeros(1, len(self.names), device=self.p.device_cfg.device)
        q[0, self.arm] = torch.as_tensor(np.asarray(q_sim) + Q_URDF_MINUS_SIM, dtype=torch.float32, device=q.device)
        return JointState.from_position(q, joint_names=self.names)

    def fk(self, q_sim):
        tp = self.p.compute_kinematics(self._state(q_sim)).tool_poses
        return T(tp.position.reshape(-1)[:3].cpu().numpy(), tp.quaternion.reshape(-1)[:4].cpu().numpy())

    def grasp_goal(self, armbase):
        return np.linalg.inv(armbase) @ self.handle @ self.handle_grasp

    def plan(self, armbase, q_start_sim, goal=None):
        """-> (trajectory [steps, 5] in sim joint angles, dt) or None."""
        from curobo.types import GoalToolPose, Pose

        goal = self.grasp_goal(armbase) if goal is None else goal
        self.p.update_world(self._scene(armbase))
        self.p.reset_seed()
        res = self.p.plan_pose(GoalToolPose.from_poses({"grasp_frame": Pose.from_list(pose7(goal))}),
                               self._state(q_start_sim), max_attempts=4)
        if res is None or not bool(res.success.any()):
            return None
        tr = res.get_interpolated_plan()
        q = tr.position.reshape(-1, len(tr.joint_names))[:, [tr.joint_names.index(j) for j in ARM]]
        q = q.cpu().numpy() - Q_URDF_MINUS_SIM
        if self.errors(goal, q[-1])[2] > self.bar_slide * 1000 + 5.0:  # 5 mm: the planner's position tolerance
            return None
        return q, float(tr.dt.reshape(-1)[0])

    def errors(self, goal, q_sim):
        """(distance from the target across the bar mm, bar misalignment deg, slide along the bar mm)."""
        err = np.linalg.inv(goal) @ self.fk(q_sim)
        slide = self.bar @ err[:3, 3]
        return (np.linalg.norm(err[:3, 3] - slide * self.bar) * 1000,
                np.degrees(np.arccos(np.clip(self.bar @ err[:3, :3] @ self.bar, -1, 1))), abs(slide) * 1000)
