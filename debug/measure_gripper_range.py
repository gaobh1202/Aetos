"""Measure the SO-ARM100 (SO101) jaw's open/close range on iris650_soarm100.

Prints tip-to-tip opening against gripper joint angle. The joint limits alone
say -10..+100 deg, which tells you nothing about what the gripper can hold;
this converts that into millimetres.

    python debug/measure_gripper_range.py

How: teleport the gripper joint to each angle (no drive, no dynamics -- a PD
target would not reach the commanded angle and the arm would sag), then take
each finger's mesh point farthest from the hinge and measure between them.
That is the "span" number a gripper is normally quoted by, and for this jaw it
is also the usable aperture because the fingers converge toward the tips.

Only the two finger meshes are used (``moving_jaw_so101_v1`` and
``wrist_roll_follower_so101_v1``); the servo body sharing the ``gripper`` link
is excluded, or it wins the farthest-point test and the number is meaningless.
"""
import argparse

from aetos_multiam import init_simulation_app

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=19, help="How many angles to sample.")
args = parser.parse_args()

simulation_app = init_simulation_app({"headless": True})

import numpy as np
import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import UsdGeom

import omni_drones.utils.kit as kit_utils
from aetos_multiam.robots.presets import get_preset
from aetos_multiam.robots.robot_base import AerialManipulatorBase

ROOT = "/World/arm"
MOVING_FINGER = f"{ROOT}/manipulator/jaw/collisions/moving_jaw_so101_v1/mesh"
FIXED_FINGER = f"{ROOT}/manipulator/gripper/collisions/wrist_roll_follower_so101_v1/mesh"
HINGE_BODY = f"{ROOT}/manipulator/jaw"

try:
    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 100,
        backend="torch", physics_prim_path="/World/PhysicsScene", device="cpu",
        sim_params={"use_gpu": False, "use_gpu_pipeline": False, "gravity": (0.0, 0.0, 0.0)},
    )
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    am = AerialManipulatorBase(get_preset("iris650_soarm100"), name="arm")
    am.spawn(translations=[(0.0, 0.0, 1.0)], prim_paths=[ROOT])
    sim.reset()
    am.initialize(prim_paths_expr=ROOT)

    view = am.drone._view
    gripper_dof = list(view.dof_names).index("gripper")
    lo, hi = [float(x) for x in view.get_dof_limits()[0, 0][gripper_dof]]

    def world_points(path):
        prim = prim_utils.get_prim_at_path(path)
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        m = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        return (m[:3, :3] @ pts.T).T + m[:3, 3]

    print(f"gripper joint limits: {lo:+.4f} .. {hi:+.4f} rad "
          f"({np.degrees(lo):+.1f} .. {np.degrees(hi):+.1f} deg)", flush=True)
    print("   q [deg]    tip-to-tip [mm]", flush=True)
    for q_val in np.linspace(lo, hi, args.steps):
        q = view.get_joint_positions().clone()
        q[..., gripper_dof] = q_val
        view.set_joint_positions(q)
        view.set_joint_velocities(torch.zeros_like(q))
        for _ in range(4):
            sim.step(render=False)
        hinge = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
            prim_utils.get_prim_at_path(HINGE_BODY)))[3, :3]
        tips = []
        for path in (MOVING_FINGER, FIXED_FINGER):
            w = world_points(path)
            tips.append(w[np.linalg.norm(w - hinge, axis=1).argmax()])
        print(f"  {np.degrees(q_val):+7.1f}    "
              f"{1000.0 * np.linalg.norm(tips[0] - tips[1]):10.1f}", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
