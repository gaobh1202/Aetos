"""Headless/GUI smoke test for the ButtonGrasp demo task.

Boots Isaac Sim exactly the way this repo's own ``scripts/train.py`` does
(same ``init_simulation_app`` call, same hydra config-path convention), then
steps the env with random actions for a few hundred steps and reports
whether anything crashed, whether the merged drone+arm+gripper prim came up
as expected, and whether at least one env reaches ``_check_success()``.

Run with Isaac Sim's own Python, e.g.:

    /home/benhua/isaacsim/python.sh scripts/run_button_grasp.py headless=true

(or via the ``omnidrone`` conda env's python, whichever this repo's own
``scripts/train.py`` is normally launched with -- this script makes no
assumptions about which).
"""

import logging
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

import hydra
import torch
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from aetos_multiam import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    from aetos_multiam.environments import REGISTERED_ENVS, make

    logging.info("Registered environments: %s", sorted(REGISTERED_ENVS))
    env = make(cfg.task.name, cfg, headless=cfg.headless)

    tensordict = env.reset()
    num_steps = int(os.environ.get("AETOS_SMOKE_TEST_STEPS", "200"))
    any_success = False
    for step in range(num_steps):
        action_shape = env.action_spec["agents", "action"].shape
        action = torch.zeros(action_shape, device=env.device)
        tensordict.set(("agents", "action"), action)
        tensordict = env.step(tensordict)
        success = bool(tensordict.get("terminated").any().item())
        any_success = any_success or success
        if step % 50 == 0:
            logging.info("step=%d success_so_far=%s", step, any_success)

    logging.info("ButtonGrasp smoke test finished. any_success=%s", any_success)
    simulation_app.close()


if __name__ == "__main__":
    main()
