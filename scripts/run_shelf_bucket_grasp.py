"""Headless/GUI smoke test for the ShelfBucketGrasp task.

Same pattern as ``run_shelf_button_grasp.py`` -- boots Isaac Sim, steps the
env with zero actions for a few hundred steps, then (in GUI mode) keeps
stepping with zero action and keeps the window open until you close it --
without ``headless=true`` the app closes itself right after the fixed step
count, which otherwise just looks like the window flashing open and shut.
Scene and robot are both set via hydra config:

    python scripts/run_shelf_bucket_grasp.py task=ShelfBucketGrasp headless=true
    python scripts/run_shelf_bucket_grasp.py task=ShelfBucketGrasp \\
        aerial_manipulators.0.preset=amov950_airbot_bottom headless=true
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
        # env.action_spec auto-unwraps to the bare per-agent leaf spec when
        # there's exactly one agent/action key (torchrl EnvBase's own
        # action_spec property getter) -- indexing it with ["agents",
        # "action"] again fails since it's no longer a CompositeSpec.
        action_shape = env.action_spec.shape
        action = torch.zeros(action_shape, device=env.device)
        tensordict.set(("agents", "action"), action)
        tensordict = env.step(tensordict)
        success = bool(tensordict.get("terminated").any().item())
        any_success = any_success or success
        if step % 50 == 0:
            logging.info("step=%d success_so_far=%s", step, any_success)

    logging.info("ShelfBucketGrasp smoke test finished. any_success=%s", any_success)

    if cfg.headless:
        simulation_app.close()
    else:
        # Keep the window open (zero action, same as the smoke test loop)
        # until the user closes it -- without this the app exits right after
        # the fixed step count above, which in GUI mode just looks like the
        # window flashing open and closing.
        logging.info("Smoke test done -- window stays open, close it (or Ctrl+C) to exit.")
        action_shape = env.action_spec.shape
        action = torch.zeros(action_shape, device=env.device)
        while simulation_app.is_running():
            tensordict.set(("agents", "action"), action)
            tensordict = env.step(tensordict)
        simulation_app.close()


if __name__ == "__main__":
    main()
