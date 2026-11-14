"""Record link poses at random arm postures, for checking a kinematic model against the simulator.

Writes aetos_multiam/data/arm_kinematics/<preset>.npz: joint names and angles, and every link's
pose (world, quaternion wxyz) at each posture. scripts/arm_planning/check_arm_kinematics.py reads it.

    export ISAACSIM_PATH=/home/benhua/isaacsim; conda activate omnidrone
    python debug/dump_arm_link_poses.py --preset uaquad600_gripper
"""
import argparse
import os.path as osp
import traceback

p = argparse.ArgumentParser()
p.add_argument("--preset", required=True)
p.add_argument("--num", type=int, default=24)
args = p.parse_args()

from aetos_multiam import init_simulation_app  # noqa: E402

simulation_app = init_simulation_app({"headless": True})
try:
    import numpy as np
    import torch
    from omni.isaac.core.simulation_context import SimulationContext

    import omni_drones.utils.kit as kit_utils

    from aetos_multiam.robots.presets import get_preset
    from aetos_multiam.robots.robot_base import AerialManipulatorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01, backend="torch",
        physics_prim_path="/World/PhysicsScene", device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
    )
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    am = AerialManipulatorBase(get_preset(args.preset), name="test_am")
    am.spawn(translations=[(0.0, 0.0, 2.0)])
    sim.reset()
    am.initialize()
    view = am.drone._view
    names = [n.split("/")[-1] for n in view.body_names]
    dof_names = list(view._dof_names)
    idx = torch.tensor([int(i) for i in am.teleop_joint_indices])
    lim = view.get_dof_limits().reshape(-1, view.num_dof, 2)[0][idx].clamp(-3.0, 3.0)
    gen = torch.Generator().manual_seed(0)
    qs, poses = [], []
    for _ in range(args.num):
        q = lim[:, 0] + (lim[:, 1] - lim[:, 0]) * (0.1 + 0.8 * torch.rand(len(idx), generator=gen))
        view.set_joint_positions(q.reshape(1, -1), joint_indices=idx)
        view.set_joint_position_targets(q.reshape(1, -1), joint_indices=idx)
        view.set_joint_velocities(torch.zeros(1, view.num_dof))
        view.set_velocities(torch.zeros(1, 6))
        sim.step(render=False)
        lt = view._physics_view.get_link_transforms().reshape(len(names), 7)
        qs.append(view.get_joint_positions().reshape(-1).numpy().copy())
        poses.append(torch.cat([lt[:, :3], lt[:, 6:7], lt[:, 3:6]], -1).numpy().copy())  # PhysX quats are xyzw
    out = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "aetos_multiam", "data", "arm_kinematics",
                   f"{args.preset}.npz")
    np.savez(out, q=np.array(qs), dof_names=np.array(dof_names), link_poses=np.array(poses), link_names=np.array(names))
    print(f"[dump] {args.preset}: {args.num} postures, links {names}, joints {dof_names} -> {out}", flush=True)
    print("BUILD done")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
