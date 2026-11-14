#!/usr/bin/env python3
"""Run the two-stage smooth-tracking curriculum end to end, unattended.

Stage 1 (`TrackAmovArmRateSmooth`) learns to track with hard action limits and a slowed
reference, but with no quadratic action penalties. Stage 2 (`TrackAmovArmRateSmooth2`)
warm-starts from stage 1 and adds the smoothness penalties.

The split is not cosmetic: quadratic action penalties push the policy's log-std down with
a gradient proportional to their weight, which overwhelms `entropy_coef` and collapses
exploration within a couple hundred iterations if applied from scratch. Stage 2 is a local
refinement of an already-working policy, so it survives them.

This script locates stage 1's checkpoint automatically, then gates stage 2 on stage 1
actually having learned to fly -- otherwise stage 2 is guaranteed to waste its budget.

Usage:
    python scripts/run_track_smooth_stages.py                     # full pipeline
    python scripts/run_track_smooth_stages.py --dry-run           # print commands only
    python scripts/run_track_smooth_stages.py \
        --stage1-checkpoint wandb/run-.../files/checkpoint_final.pt   # stage 2 only
    python scripts/run_track_smooth_stages.py -- task.env.num_envs=256

Anything after a bare `--` is appended verbatim as Hydra overrides to both stages.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Recommended algo settings, from the post-mortem of run uiqo8rm0 (collapsed) vs
# v0vlcuht (worked). entropy_coef stays at its 0.001 default -- the log-std bounds are
# what make stage 2's penalties safe, and the ceiling keeps sigma from drifting to ~0.9
# late in training, where the sampled action is mostly noise.
STAGE1_ALGO = [
    "algo.actor_log_std_init=-1.0",
    "algo.actor_log_std_min=-1.6",
    "algo.actor_log_std_max=-0.8",
]
STAGE2_ALGO = [
    "algo.actor_log_std_min=-1.6",
    "algo.actor_log_std_max=-0.8",
]

# Stage-1 health gate. The two reference points, both at 30M frames:
#   collapsed run: episode_len 72/1000 (7%), tracking_error_ema 1.24, sigma 0.043
#   working run:   episode_len 563/600 (94%), tracking_error_ema 0.203, sigma 0.92
GATE_MIN_EPLEN_FRAC = 0.40
GATE_MAX_TRACKING_ERROR_EMA = 1.0
GATE_MIN_SIGMA = 0.10

# Fields that determine the policy's input/output shapes. If these differ between the two
# stages, `load_state_dict(..., strict=False)` silently drops the mismatched layers and
# stage 2 trains from near-scratch *with* the penalties enabled -- i.e. it collapses.
SHAPE_CRITICAL_FIELDS = [
    "future_traj_steps",
    "time_encoding",
    "intrinsics",
    "action_transform",
    "drone_model.name",
    "drone_model.controller",
]


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"[pipeline] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def compose_task_cfg(task: str):
    """Compose a task config the same way train.py does, without starting Isaac Sim."""
    import importlib

    importlib.import_module("omni_drones.learning.ppo.ppo")  # registers the `algo/ppo` group
    from hydra import compose, initialize_config_dir

    os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(REPO / "cfg"))
    with initialize_config_dir(config_dir=str(REPO / "scripts"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    return cfg


def get_nested(cfg, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def preflight(args) -> dict:
    """Validate the environment and both task configs before burning any GPU hours."""
    if not (REPO / "scripts" / "train.py").is_file():
        die(f"could not find scripts/train.py under {REPO}")

    usd = Path(args.usd_path)
    if not usd.is_file():
        die(
            f"AmovDrone USD not found: {usd}\n"
            "         pass --usd-path, or set OMNIDRONES_AMOV_USD_PATH"
        )

    for task in (args.stage1_task, args.stage2_task):
        if not (REPO / "cfg" / "task" / f"{task}.yaml").is_file():
            die(f"task config not found: cfg/task/{task}.yaml")

    log("composing task configs to check stage compatibility ...")
    try:
        cfg1 = compose_task_cfg(args.stage1_task)
        cfg2 = compose_task_cfg(args.stage2_task)
    except Exception as exc:  # noqa: BLE001 - surface the hydra error verbatim
        die(f"failed to compose task configs: {exc}")

    mismatched = []
    for field in SHAPE_CRITICAL_FIELDS:
        try:
            v1, v2 = get_nested(cfg1.task, field), get_nested(cfg2.task, field)
        except Exception:  # noqa: BLE001 - field absent in one of them is itself a mismatch
            mismatched.append((field, "<missing>", "<missing>"))
            continue
        if v1 != v2:
            mismatched.append((field, v1, v2))
    if mismatched:
        detail = "\n".join(f"           {f}: stage1={a!r} stage2={b!r}" for f, a, b in mismatched)
        die(
            "stage 1 and stage 2 disagree on fields that change the policy's tensor "
            f"shapes:\n{detail}\n"
            "         The stage-2 warm start would silently discard those weights."
        )

    max_ep_len = int(cfg1.task.env.max_episode_length)
    action_dim = 4 if str(cfg1.task.action_transform) in ("rate", "attitude") else None
    log(f"configs compatible (max_episode_length={max_ep_len}, action_dim={action_dim})")
    return {"max_episode_length": max_ep_len, "action_dim": action_dim}


def build_cmd(task: str, frames: int, algo_overrides: list[str], args) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        f"task={task}",
        "algo=ppo",
        f"headless={str(args.headless).lower()}",
        f"total_frames={frames}",
        f"save_interval={args.save_interval}",
        f"seed={args.seed}",
    ]
    cmd += algo_overrides
    cmd += args.extra
    return cmd


def run_stage(name: str, cmd: list[str], log_path: Path, args) -> int:
    """Run a training stage, streaming its output to both the console and a log file."""
    log(f"{name}: {' '.join(shlex.quote(c) for c in cmd)}")
    log(f"{name}: log -> {log_path}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["OMNIDRONES_AMOV_USD_PATH"] = args.usd_path

    started = datetime.datetime.now()
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        with open(log_path, "wb") as lf:
            while True:
                # read1 returns as soon as any bytes are available, so tqdm's
                # carriage-return progress updates come through live.
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                lf.write(chunk)
                lf.flush()
        ret = proc.wait()
    except KeyboardInterrupt:
        log(f"{name}: interrupted, terminating child ...")
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    elapsed = datetime.datetime.now() - started
    log(f"{name}: exit={ret} elapsed={str(elapsed).split('.')[0]}")
    return ret


def find_checkpoint(log_path: Path, known_runs: set[str]) -> Path | None:
    """Locate the checkpoint stage 1 just wrote.

    Primary source is train.py's own "Saved checkpoint to <path>" line, which is exact.
    Falls back to whichever wandb run directory appeared while stage 1 was running --
    `wandb.run_id` cannot be used to pin the directory in advance because init_wandb
    treats it as a resume request.
    """
    if log_path.is_file():
        text = log_path.read_text(errors="ignore")
        hits = re.findall(r"Saved checkpoint to (\S*checkpoint_final\.pt)", text)
        if hits:
            path = Path(hits[-1])
            if not path.is_absolute():
                path = REPO / path
            if path.is_file():
                return path

    candidates = [
        p
        for pattern in ("run-*", "offline-run-*")  # offline mode prefixes the directory
        for p in (REPO / "wandb").glob(pattern)
        if p.is_dir() and not p.is_symlink() and p.name not in known_runs
    ]
    new_runs = sorted(candidates, key=lambda p: p.stat().st_mtime)
    for run_dir in reversed(new_runs):
        for candidate in ("files/checkpoint_final.pt", "files/checkpoint_latest.pt"):
            path = run_dir / candidate
            if path.is_file():
                return path
    return None


def sigma_from_entropy(entropy: float, action_dim: int) -> float:
    # Diagonal Gaussian: H = d * (0.5*ln(2*pi*e) + ln sigma)
    return math.exp(entropy / action_dim - 0.5 * math.log(2 * math.pi * math.e))


def read_stage1_metrics(ckpt_path: Path, log_path: Path) -> dict:
    """Pull the final metrics for the health gate, preferring wandb's own summary."""
    metrics: dict = {}
    summary = ckpt_path.parent / "wandb-summary.json"
    if summary.is_file():
        try:
            metrics = json.loads(summary.read_text())
        except json.JSONDecodeError:
            metrics = {}

    if not metrics and log_path.is_file():
        # Fall back to the last printed train block. train.py prints one YAML block per
        # iteration; the final eval is only sent to wandb, not stdout.
        pat = re.compile(r"^([a-zA-Z_][\w./]*): (-?[\d.]+(?:e[+-]?\d+)?)$")
        for line in log_path.read_text(errors="ignore").splitlines():
            m = pat.match(line.strip())
            if m:
                metrics[m.group(1)] = float(m.group(2))
    return metrics


def check_gate(metrics: dict, info: dict) -> tuple[bool, list[str]]:
    """Decide whether stage 1 produced a policy worth fine-tuning."""
    max_ep = info["max_episode_length"]
    action_dim = info["action_dim"]

    def pick(*keys):
        for k in keys:
            if k in metrics:
                return k, metrics[k]
        return None, None

    lines: list[str] = []
    verdicts: list[bool] = []

    key, ep_len = pick("eval/stats.episode_len", "train/stats.episode_len")
    if ep_len is None:
        lines.append("  episode_len              : UNKNOWN (metric not found)")
        verdicts.append(False)
    else:
        frac = ep_len / max_ep
        ok = frac >= GATE_MIN_EPLEN_FRAC
        verdicts.append(ok)
        lines.append(
            f"  {key:25s}: {ep_len:8.1f} / {max_ep} = {frac:5.1%}   "
            f"[{'PASS' if ok else 'FAIL'}, need >= {GATE_MIN_EPLEN_FRAC:.0%}]"
        )

    key, trk = pick("eval/stats.tracking_error_ema", "train/stats.tracking_error_ema")
    if trk is None:
        lines.append("  tracking_error_ema       : UNKNOWN (metric not found)")
        verdicts.append(False)
    else:
        ok = trk <= GATE_MAX_TRACKING_ERROR_EMA
        verdicts.append(ok)
        lines.append(
            f"  {key:25s}: {trk:8.3f}                   "
            f"[{'PASS' if ok else 'FAIL'}, need <= {GATE_MAX_TRACKING_ERROR_EMA}]"
        )

    if action_dim and "entropy" in metrics:
        sigma = sigma_from_entropy(metrics["entropy"], action_dim)
        ok = sigma >= GATE_MIN_SIGMA
        verdicts.append(ok)
        lines.append(
            f"  {'sigma (from entropy)':25s}: {sigma:8.3f}                   "
            f"[{'PASS' if ok else 'FAIL'}, need >= {GATE_MIN_SIGMA}]"
        )

    return (bool(verdicts) and all(verdicts)), lines


def main() -> int:
    default_usd = os.environ.get(
        "OMNIDRONES_AMOV_USD_PATH", str(REPO / "amov_drone_arm_1.usd")
    )
    p = argparse.ArgumentParser(
        description="Run the stage-1 -> stage-2 smooth-tracking curriculum unattended.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage1-task", default="TrackAmovArmRateSmooth")
    p.add_argument("--stage2-task", default="TrackAmovArmRateSmooth2")
    p.add_argument("--stage1-frames", type=int, default=30_000_000,
                   help="the working baseline converged within 30M (default: %(default)s)")
    p.add_argument("--stage2-frames", type=int, default=15_000_000,
                   help="fine-tuning needs less than stage 1 (default: %(default)s)")
    p.add_argument("--stage1-checkpoint", default=None,
                   help="skip stage 1 and fine-tune this checkpoint directly")
    p.add_argument("--skip-stage2", action="store_true", help="run stage 1 only")
    p.add_argument("--skip-gate", action="store_true",
                   help="run stage 2 even if stage 1 looks collapsed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--usd-path", default=default_usd)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True,
                   help="headless training is ~faster; --no-headless opens the viewer")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("extra", nargs="*", default=[],
                   help="extra Hydra overrides, applied to both stages (put after --)")
    args = p.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else REPO / "experiments_results" / f"track_smooth_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log(f"repo      : {REPO}")
    log(f"log dir   : {log_dir}")
    log(f"usd       : {args.usd_path}")

    info = preflight(args)

    # ---------------- stage 1 ----------------
    if args.stage1_checkpoint:
        ckpt = Path(args.stage1_checkpoint)
        if not ckpt.is_absolute():
            ckpt = REPO / ckpt
        if not ckpt.is_file() and not args.dry_run:
            die(f"--stage1-checkpoint does not exist: {ckpt}")
        log(f"stage 1   : SKIPPED, fine-tuning {ckpt}")
        stage1_log = Path("/dev/null")
    else:
        known_runs = {p.name for p in (REPO / "wandb").glob("run-*")}
        stage1_log = log_dir / "stage1.log"
        cmd = build_cmd(args.stage1_task, args.stage1_frames, STAGE1_ALGO, args)
        ret = run_stage("stage 1", cmd, stage1_log, args)
        if ret != 0:
            die(f"stage 1 exited with code {ret}; not starting stage 2. See {stage1_log}")

        if args.dry_run:
            ckpt = Path("<stage1-checkpoint>")
        else:
            found = find_checkpoint(stage1_log, known_runs)
            if found is None:
                die(
                    "stage 1 finished but no checkpoint was found.\n"
                    f"         Grep for 'Saved checkpoint to' in {stage1_log}, then rerun "
                    "with --stage1-checkpoint <path>."
                )
            ckpt = found
            log(f"stage 1 checkpoint: {ckpt}")

    if args.skip_stage2:
        log("--skip-stage2 given; stopping after stage 1.")
        return 0

    # ---------------- health gate ----------------
    if not args.dry_run and not args.stage1_checkpoint:
        metrics = read_stage1_metrics(ckpt, stage1_log)
        passed, lines = check_gate(metrics, info)
        log("stage 1 health check:")
        for line in lines:
            print(line, flush=True)
        if not passed:
            if args.skip_gate:
                log("gate FAILED but --skip-gate given; continuing into stage 2 anyway.")
            else:
                die(
                    "stage 1 did not produce a tracking policy, so stage 2 would waste its "
                    "budget.\n"
                    "         The smoothness penalties only work as a refinement of a policy "
                    "that already flies.\n"
                    f"         Inspect {stage1_log} (see the expected learning curve in "
                    f"cfg/task/{args.stage1_task}.yaml), then either retrain stage 1 or "
                    "rerun with --skip-gate."
                )
        else:
            log("stage 1 health check PASSED")

    # ---------------- stage 2 ----------------
    stage2_log = log_dir / "stage2.log"
    cmd = build_cmd(
        args.stage2_task, args.stage2_frames,
        STAGE2_ALGO + [f"algo.checkpoint_path={ckpt}"], args,
    )
    ret = run_stage("stage 2", cmd, stage2_log, args)
    if ret != 0:
        die(f"stage 2 exited with code {ret}. See {stage2_log}")

    if not args.dry_run:
        final = find_checkpoint(stage2_log, set())
        log(f"stage 2 checkpoint: {final if final else '<not found>'}")
    log("pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
