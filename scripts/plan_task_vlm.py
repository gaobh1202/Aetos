"""Task allocation and skill scheduling with a VLM: scene (+ image) -> plan -> execution program.

    python scripts/plan_task_vlm.py --scene outputs/iris650_soarm100_AerialTransport_planning/scene.json --image outputs/iris650_soarm100_AerialTransport_planning/scene.png
    python scripts/plan_task_vlm.py --scene outputs/iris650_soarm100_AerialTransport_planning/scene.json --reference      # hand-written plan, no VLM

Needs only numpy and, for the VLM, the `openai` and `jsonschema` packages and an API key (store it once with
scripts/set_vlm_key.py); it does not need Isaac Sim. The endpoint is OpenAI-compatible: --model / --base-url, or
AETOS_VLM_MODEL / AETOS_VLM_BASE_URL (default: qwen4-max). scripts/export_transport_scene.py writes the scene files.
Writes plan.json (what the planner decided), program.json (resolved poses, reference segments, schedule) and
attempts.json (every VLM round, with the validator's objections) next to the scene, or to --out.
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Load aetos_multiam.planning without importing aetos_multiam itself (which needs Isaac Sim).
pkg = types.ModuleType("aetos_planning")
pkg.__path__ = [str(REPO / "aetos_multiam/planning")]
sys.modules["aetos_planning"] = pkg
from aetos_planning.compile import compile_plan  # noqa: E402
from aetos_planning.plan import validate_plan  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--scene", required=True)
p.add_argument("--image", default=None, help="PNG of the scene (omit: text and poses only)")
p.add_argument("--instruction", default=None, help="overrides the scene file's instruction")
p.add_argument("--reference", action="store_true", help="use the hand-written transport plan instead of the VLM")
p.add_argument("--model", default=None)
p.add_argument("--base-url", default=None)
p.add_argument("--out", default=None, help="output directory (default: the scene file's)")
args = p.parse_args()

scene = json.loads(Path(args.scene).read_text())
if args.instruction:
    scene["instruction"] = args.instruction
out = Path(args.out or Path(args.scene).parent)
out.mkdir(parents=True, exist_ok=True)
if args.reference:
    from aetos_planning.reference_plans import transport_plan

    plan, attempts = transport_plan(), []
    errors = validate_plan(plan, scene)
    if errors:
        raise SystemExit("reference plan rejected:\n- " + "\n- ".join(errors))
else:
    from aetos_planning import vlm

    plan, attempts = vlm.plan_with_vlm(scene, args.image, model=args.model, base_url=args.base_url)
    (out / "attempts.json").write_text(json.dumps(attempts, indent=1))
    print(f"[plan] {len(attempts)} VLM round(s), {sum(a['input_tokens'] for a in attempts)} input / "
          f"{sum(a['output_tokens'] for a in attempts)} output tokens")
program = compile_plan(plan, scene)
(out / "plan.json").write_text(json.dumps(plan, indent=1))
(out / "program.json").write_text(json.dumps(program, indent=1))
print("[plan] reasoning:", plan["reasoning"])
for a in plan["allocation"]:
    print(f"[plan] {a['robot']} -> {a['part']}: {a['reason']}")
for e in program["subtasks"]:
    d = e["drone"]
    print(f"[plan] {e['t_start']:5.1f}-{e['t_end']:5.1f}s {e['robot']} {e['id']:14s} drone {d['skill']:5s} ({d['mode']}) "
          f"arm {e['arm']['skill']:7s} until {e['done_when']:8s} after {e['deps']}")
print(f"[plan] nominal duration {program['duration_s']:.1f} s -> {out}/plan.json, program.json")
