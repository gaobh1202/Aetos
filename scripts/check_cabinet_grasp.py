"""Assert that AerialCabinetGrasp's cabinet is the fixture the task assumes.

No robot involved. Checks, against the live simulation:

* the doors start shut, and ``reset_doors`` puts them where it is asked to;
* the carcass does not move when a door is swung (it is welded to the world);
* ``Cabinet.handle_position`` -- the analytic point every waypoint in the task
  is built from -- agrees with where the door body actually carries its handle,
  shut, half open and fully open;
* a door pushed on its handle opens, i.e. the hinge is free and the handle's
  outward normal points the way the geometry says;
* the button is standing on its shelf, inside the cabinet.

    python scripts/check_cabinet_grasp.py task=AerialCabinetGrasp headless=true
    python scripts/check_cabinet_grasp.py task=AerialCabinetGrasp headless=true \
        +screenshot_dir=resources/scenes

Exits non-zero with the failing measurement if anything is off.
"""
import logging
import math
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

import hydra
import torch
from omegaconf import OmegaConf

SETTLE_STEPS = 120
HANDLE_TOLERANCE_M = 0.01
PUSH_FORCE_N = 8.0
PUSH_STEPS = 250


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    screenshot_dir = cfg.get("screenshot_dir", None)
    if cfg.task.name != "AerialCabinetGrasp":
        raise SystemExit("run with task=AerialCabinetGrasp (got task=%s)" % cfg.task.name)
    cfg.task.env.num_envs = 1

    from aetos_multiam import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    from omni_drones.utils.torch import quat_rotate

    from aetos_multiam.environments import make

    env = make(cfg.task.name, cfg, headless=cfg.headless)
    env.reset()
    cabinet = env.cabinet
    view = cabinet._view
    origin = env.envs_positions[0]
    all_envs = torch.arange(env.num_envs, device=env.device)
    body_names = [n.split("/")[-1] for n in view.body_names]
    ok = True

    def settle(steps=SETTLE_STEPS):
        for _ in range(steps):
            env.sim.step(render=False)

    def link_pose(name):
        """Env-frame position and (w, x, y, z) quaternion of one cabinet body."""
        t = view._physics_view.get_link_transforms().reshape(-1, len(body_names), 7)[0]
        t = t[body_names.index(name)]
        return t[:3] - origin, torch.cat([t[6:7], t[3:6]])      # PhysX quats are xyzw

    def live_handle(door):
        """Where the door body carries its handle right now: the handle's
        offset from the door's own origin, as authored, moved with the body."""
        ox, oy = cabinet.door_origin_xy[door]
        px, py, pz = cabinet.handle_center_open[door]
        pos, quat = link_pose(cabinet.door_body_names[door])
        local = torch.tensor([px - ox, py - oy, 0.0], device=env.device) * cabinet.scale
        p = pos + quat_rotate(quat.unsqueeze(0), local.unsqueeze(0))[0]
        p[2] = cabinet.translation[2] + cabinet.scale * pz
        return p

    def shot(name, eye, target):
        if not screenshot_dir:
            return
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf

        vp = get_active_viewport()
        cs = ViewportCameraState("/OmniverseKit_Persp", vp)
        o = [float(v) for v in origin]
        cs.set_position_world(Gf.Vec3d(o[0] + eye[0], o[1] + eye[1], eye[2]), True)
        cs.set_target_world(Gf.Vec3d(o[0] + target[0], o[1] + target[1], target[2]), True)
        for _ in range(40):
            env.sim.render()
        out = Path(screenshot_dir)
        out.mkdir(parents=True, exist_ok=True)
        capture_viewport_to_file(vp, str(out / name))
        for _ in range(10):
            env.sim.render()
        logging.info("saved %s", out / name)

    # -- robots face the cabinet ---------------------------------------------
    # Taken before anything is stepped: nothing is flying the robots here, so
    # this is the only moment they are still where the task spawned them.
    shot("AerialCabinetGrasp.png", (4.6, -7.4, 4.4), (0.0, 0.2, 1.1))
    cabinet_xy = torch.tensor(cabinet.translation[:2], device=env.device)
    for i, am in enumerate(env.aerial_manipulators):
        state = am.drone.get_state()[0, 0]
        heading = quat_rotate(state[3:7].unsqueeze(0),
                              torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0, :2]
        to_cabinet = cabinet_xy - (state[:2] - origin[:2])
        cos = float(torch.dot(heading, to_cabinet) / (heading.norm() * to_cabinet.norm()))
        logging.info("agent %d spawns at %s, heading %.0f deg off the cabinet", i,
                     [round(float(v), 2) for v in state[:3] - origin], math.degrees(math.acos(cos)))
        if cos < math.cos(math.radians(30.0)):
            logging.error("agent %d does NOT start facing the cabinet", i)
            ok = False

    # -- doors start shut ----------------------------------------------------
    settle()
    frac = cabinet.get_door_open_fraction()[0]
    logging.info("after reset: door open fraction = %s", [round(float(f), 4) for f in frac])
    if float(frac.max()) > 0.02:
        logging.error("doors are NOT shut after reset")
        ok = False
    carcass_start, _ = link_pose(cabinet.body_name)

    # -- analytic handle position against the simulated door -----------------
    for target in (0.0, 0.5, 1.0):
        cabinet.reset_doors(all_envs, open_fraction=target)
        settle()
        got = cabinet.get_door_open_fraction()[0]
        for door in range(2):
            want = torch.tensor(cabinet.handle_position(door, target), device=env.device)
            live = live_handle(door)
            err = float(torch.norm(want - live))
            logging.info("door %d at %.1f open (read back %.3f): handle analytic %s  live %s  err %.4f m",
                         door, target, float(got[door]),
                         [round(float(v), 3) for v in want], [round(float(v), 3) for v in live], err)
            if abs(float(got[door]) - target) > 0.03:
                logging.error("door %d did not hold at %.1f open", door, target)
                ok = False
            if err > HANDLE_TOLERANCE_M * cabinet.scale:
                logging.error("door %d: handle_position is off by %.4f m at %.1f open", door, err, target)
                ok = False
    shot("AerialCabinetGrasp_open.png", (1.9, -4.6, 2.6), (0.0, 0.9, 1.2))

    # -- carcass welded ------------------------------------------------------
    carcass_end, _ = link_pose(cabinet.body_name)
    drift = float(torch.norm(carcass_end - carcass_start))
    logging.info("carcass drift over the door sweeps: %.5f m", drift)
    if drift > 1e-3:
        logging.error("the cabinet carcass moved -- it is not welded to the world")
        ok = False

    # -- a pull on the handle opens the door ---------------------------------
    cabinet.reset_doors(all_envs, open_fraction=0.0)
    settle(20)
    num_links = len(body_names)
    for _ in range(PUSH_STEPS):
        forces = torch.zeros(1, num_links, 3, device=env.device)
        positions = torch.zeros(1, num_links, 3, device=env.device)
        for door in range(2):
            link = body_names.index(cabinet.door_body_names[door])
            frac_now = float(cabinet.get_door_open_fraction()[0, door])
            nx, ny = cabinet.handle_normal(door, frac_now)
            forces[0, link] = torch.tensor([nx, ny, 0.0], device=env.device) * PUSH_FORCE_N
            positions[0, link] = live_handle(door) + origin
        view._physics_view.apply_forces_and_torques_at_position(
            forces.reshape(-1, 3), None, positions.reshape(-1, 3),
            torch.arange(view.count, device=env.device), True)
        env.sim.step(render=False)
    frac = cabinet.get_door_open_fraction()[0]
    logging.info("after %.0f N along each handle's normal for %d steps: open fraction = %s",
                 PUSH_FORCE_N, PUSH_STEPS, [round(float(f), 3) for f in frac])
    if float(frac.min()) < env.door_open_threshold:
        logging.error("a door did NOT open under a pull on its handle")
        ok = False

    # -- button on its shelf -------------------------------------------------
    want = torch.tensor(env.button_position, device=env.device)
    got = env.button.get_world_pose()[0].reshape(-1, 3)[0] - origin
    width, depth, height = cabinet.compartment_clear_size(env.button_compartment)
    logging.info("button: expected %s, resting at %s; its compartment is %.2f wide, %.2f deep, %.2f high",
                 [round(float(v), 3) for v in want], [round(float(v), 3) for v in got], width, depth, height)
    if float(torch.norm(got[:2] - want[:2])) > 0.02 or abs(float(got[2] - want[2])) > 0.01:
        logging.error("the button is NOT resting where the task says it is")
        ok = False

    for i in range(env.num_agents):
        logging.info("agent %d goal: %s", i, [round(float(v), 3) for v in env._goal_position(i)])

    if ok:
        logging.info("Cabinet check PASSED.")
    else:
        logging.error("Cabinet check FAILED -- see the measurements above.")
    env.close()
    if not ok:
        # Not SystemExit after simulation_app.close(): Kit ends the process
        # inside close() with status 0, so a raise after it never runs.
        logging.shutdown()
        os._exit(1)
    simulation_app.close()


if __name__ == "__main__":
    main()
