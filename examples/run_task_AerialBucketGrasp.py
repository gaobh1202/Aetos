"""Run AerialBucketGrasp as an actual task, not just the zero-action smoke
test ``scripts/run_aerial_bucket_grasp.py`` already is.

Unlike ``run_task_AerialButtonGrasp.py``, this doesn't have a policy+replay
pair to drive it: AerialBucketGrasp is a brand-new task with no trained
hover/reach policy and no recorded raids-am-simulator episode for this arm
to replay. The drone side needs nothing anyway -- it's welded to the world
at spawn (``AerialBucketGrasp``'s own design, see that class's module
docstring), so there's no drone control source to plug in here at all.

For the arm, this script does a brief **random-shooting search** instead of
real inverse kinematics (which would need deriving/validating a kinematic
chain per preset -- overkill for "make the scene visibly do something"):
sample ``--search-trials`` random normalized joint targets, give each
``--search-settle-steps`` physics steps to settle from a fresh reset, score
it by the env's own ``rpos_eef_bucket`` norm (the same quantity
``AerialBucketGrasp.reward()`` shapes on), and keep the closest. The arm is
then smoothly ramped from its reset pose to that best-found target over
``--ramp-steps`` and held there -- enough real motion to see the task scene
actually run end to end, without claiming to be a real reach policy.

Usage:
    python examples/run_task_AerialBucketGrasp.py task=AerialBucketGrasp
    python examples/run_task_AerialBucketGrasp.py task=AerialBucketGrasp headless=true --max-steps 500
    python examples/run_task_AerialBucketGrasp.py task=AerialBucketGrasp --search-trials 80
"""

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

# argparse for this script's own flags, hydra for the usual task=/headless=
# overrides -- split the argv so hydra never sees (and chokes on) --search-trials
# etc., same reasoning amteleop_soarm_leader.py's device-arg split uses.
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--search-trials", type=int, default=40)
_parser.add_argument("--search-settle-steps", type=int, default=20)
_parser.add_argument("--ramp-steps", type=int, default=60)
_parser.add_argument("--max-steps", type=int, default=300, help="0 = run until the window closes / Ctrl+C.")
_own_args, _hydra_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _hydra_argv

import hydra
import torch
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from aetos_multiam import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    from aetos_multiam.environments import make

    env = make(cfg.task.name, cfg, headless=cfg.headless)
    device = env.device

    def distance_now() -> float:
        return torch.norm(env.rpos_eef_bucket, dim=-1).mean().item()

    # -- search for a decent static "reach" pose ----------------------------
    # First half: pure random shooting over the full [-1, 1] action range,
    # for coverage. Second half: local refinement -- perturb the current
    # best with noise that decays linearly to 0, i.e. simple hill-climbing --
    # a joint-space arm (so_arm100 has 6 DOF) is too high-dimensional for
    # pure random shooting alone to land close in a handful of trials.
    action_shape = env.action_spec.shape
    best_action = torch.zeros(action_shape, device=device)
    best_distance = float("inf")
    num_random = max(1, _own_args.search_trials // 2)

    for trial in range(_own_args.search_trials):
        tensordict = env.reset()
        if trial < num_random:
            candidate = torch.empty(action_shape, device=device).uniform_(-1.0, 1.0)
        else:
            refine_frac = (trial - num_random) / max(1, _own_args.search_trials - num_random)
            noise_scale = 0.5 * (1.0 - refine_frac)
            noise = torch.empty(action_shape, device=device).uniform_(-1.0, 1.0) * noise_scale
            candidate = (best_action + noise).clamp(-1.0, 1.0)
        for _ in range(_own_args.search_settle_steps):
            tensordict.set(("agents", "action"), candidate)
            tensordict = env.step(tensordict)
        distance = distance_now()
        if distance < best_distance:
            best_distance = distance
            best_action = candidate.clone()
        logging.info(
            "[search] trial=%d distance=%.4f best_so_far=%.4f", trial, distance, best_distance
        )

    logging.info(
        "[run_task_AerialBucketGrasp] search done, best_distance=%.4f after %d trials",
        best_distance, _own_args.search_trials,
    )

    # -- ramp from the reset pose to the best-found pose, then hold --------
    tensordict = env.reset()
    step = 0
    for ramp_step in range(_own_args.ramp_steps):
        action = best_action * ((ramp_step + 1) / _own_args.ramp_steps)
        tensordict.set(("agents", "action"), action)
        tensordict = env.step(tensordict)
        step += 1

    logging.info(
        "[run_task_AerialBucketGrasp] reach ramp done at step=%d distance=%.4f", step, distance_now()
    )

    any_success = False
    try:
        while simulation_app.is_running():
            tensordict.set(("agents", "action"), best_action)
            tensordict = env.step(tensordict)
            step += 1
            success = bool(tensordict.get("terminated").any().item())
            any_success = any_success or success
            if step % 50 == 0:
                logging.info(
                    "[run_task_AerialBucketGrasp] step=%d distance=%.4f any_success=%s",
                    step, distance_now(), any_success,
                )
            if _own_args.max_steps and step >= _own_args.max_steps:
                break
    except KeyboardInterrupt:
        pass

    logging.info(
        "[run_task_AerialBucketGrasp] finished at step=%d distance=%.4f any_success=%s",
        step, distance_now(), any_success,
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
