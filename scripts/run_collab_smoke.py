"""Generic headless smoke test for any collaborative (multi-agent) task,
e.g. AerialTransport / AerialAssemblyTriple -- mirrors run_button_grasp.py
but handles the per-agent action-spec keys (f"agent_{i}", "action") that
CollaborativeAerialEnv uses instead of ButtonGrasp's single "agents" key.

    python scripts/run_collab_smoke.py task=AerialTransport headless=true task.env.num_envs=2
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
    num_steps = int(os.environ.get("AETOS_SMOKE_TEST_STEPS", "50"))
    for step in range(num_steps):
        for key in env._agent_keys:
            action_shape = env.action_spec[key, "action"].shape
            tensordict.set((key, "action"), torch.zeros(action_shape, device=env.device))
        tensordict = env.step(tensordict)
        if step % 10 == 0:
            logging.info("step=%d ok", step)

    logging.info("Collab smoke test (%s) finished OK.", cfg.task.name)
    simulation_app.close()


if __name__ == "__main__":
    main()
