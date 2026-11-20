"""Ask a vision-language model for a plan: scene image + object poses + instruction -> plan (PLAN_SCHEMA).

Talks to an OpenAI-compatible chat endpoint (default: Qwen). Settings, in order of precedence:
arguments, then the environment (AETOS_VLM_MODEL, AETOS_VLM_BASE_URL, AETOS_VLM_API_KEY), then the key file
written by scripts/set_vlm_key.py. What leaves the machine: the instruction, the scene description (names
and poses) and the image.
"""
import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .plan import ARM_SKILLS, DONE_WHEN, DRONE_SKILLS, PLAN_SCHEMA, SITES, SPEEDS, validate_plan

MODEL = "qwen4-max"
BASE_URL = "https://token-plan.maas.qwencloudapi.com/compatible-mode/v1"
KEY_FILE = Path.home() / ".config" / "aetos" / "vlm_api_key"


def api_key() -> str:
    key = os.environ.get("AETOS_VLM_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        raise RuntimeError(f"no API key: run scripts/set_vlm_key.py (writes {KEY_FILE}) or set AETOS_VLM_API_KEY")
    return key


def system_prompt() -> str:
    def table(d):
        return "\n".join(f"- {k}: {v}" for k, v in d.items())

    return f"""You are the task planner of a team of aerial manipulators: quadrotors that each carry a small \
robot arm with a parallel gripper. You receive an instruction, a description of the scene (names and poses of \
the robots and objects, world frame, metres and degrees, z up) and usually an image of it. You decide which \
robot does what, in which order, and with which skill. You do not produce coordinates or trajectories: \
low-level policies execute each subtask, and a geometry module turns the named sites into poses.

Why the structure matters: each drone is flown by a learned controller that tracks a reference point, and each \
arm is driven by a learned visuomotor policy that only succeeds when its drone holds still at the right place \
next to the object. So a robot first flies to a site, then hovers there while its arm works, then moves on. \
An object too large for one robot is held by several and moved in formation.

Drone skills (one per subtask):
{table(DRONE_SKILLS)}

Arm skills (one per subtask, running while the drone skill runs):
{table(ARM_SKILLS)}

Sites (where the drone skill is aimed; `reference` names the object or part the site belongs to):
{table(SITES)}

A subtask ends on `done_when`:
{table(DONE_WHEN)}

Speed is one of {", ".join(SPEEDS)}: fast for free flight far from anything, normal near objects, slow when \
approaching a handle and whenever an object is carried.

Rules the executor enforces:
- A subtask belongs to one robot. A robot's subtasks run in the order listed; `after` adds waits on other \
robots' subtasks. List subtasks so that everything named in `after` appears earlier.
- Approach a handle through its pregrasp_base, then goto its grasp_base, then hover there with arm skill grasp \
(arm_object = that handle, done_when = grasped).
- Once a robot holds a handle its arm skill is hold until it releases, and its drone may only hover or carry.
- carry: every robot holding the object takes part, one subtask each, all with the same sync_group, the same \
site and reference, arm skill hold and arm_object = the handle that robot holds. Lift first (site lifted, \
reference = the object), then move it above the destination, then set it down on it.
- A jointly held object must not be lifted before every holder has grasped: make the lift wait (`after`) for \
all the grasps.
- release: drone hovering at the site it reached, arm skill release, done_when = released. Then retreat \
from that handle, then go home.
- Assign each handle to the robot that reaches it with the least crossing of other robots' paths; say why in \
`allocation`.

Use the object and part names exactly as given in the scene description.

Answer with one JSON object and nothing else. It must conform to this JSON Schema (every listed property is \
required; use "" or [] where a field does not apply):
{json.dumps(PLAN_SCHEMA)}"""


def _user_content(scene: dict, image: Optional[Path]) -> list:
    content = []
    if image is not None:
        data = base64.standard_b64encode(Path(image).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
    content.append({"type": "text", "text": "Instruction: " + scene["instruction"] + "\n\nScene:\n"
                    + json.dumps({k: scene[k] for k in ("robots", "objects")}, indent=1)
                    + "\n\nProduce the plan as JSON."})
    return content


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):  # some models fence their JSON despite the instruction
        text = text.split("```")[1].removeprefix("json")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def _schema_errors(plan) -> List[str]:
    import jsonschema

    v = jsonschema.Draft202012Validator(PLAN_SCHEMA)
    return [f"at {'/'.join(map(str, e.absolute_path)) or 'top level'}: {e.message}" for e in v.iter_errors(plan)]


def plan_with_vlm(scene: dict, image: Optional[Path] = None, model: Optional[str] = None,
                  base_url: Optional[str] = None, repairs: int = 2) -> Tuple[dict, List[dict]]:
    """Returns (plan, attempts). A plan that is malformed or that the validator rejects is sent back with the
    objections, up to `repairs` times; attempts records every round (plan, errors, token usage)."""
    import openai

    model = model or os.environ.get("AETOS_VLM_MODEL", MODEL)
    client = openai.OpenAI(api_key=api_key(), base_url=base_url or os.environ.get("AETOS_VLM_BASE_URL", BASE_URL))
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": _user_content(scene, image)}]
    attempts: List[dict] = []
    for _ in range(repairs + 1):
        resp = client.chat.completions.create(model=model, messages=messages, response_format={"type": "json_object"})
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = dict(model=resp.model, input_tokens=getattr(resp.usage, "prompt_tokens", 0),
                     output_tokens=getattr(resp.usage, "completion_tokens", 0))
        if choice.finish_reason == "length":
            raise RuntimeError("the plan was cut off (finish_reason = length)")
        try:
            plan = _parse(text)
            errors = _schema_errors(plan)
            errors = errors or validate_plan(plan, scene)  # the semantic checks assume a well-formed plan
        except (ValueError, json.JSONDecodeError) as e:
            plan, errors = None, [f"the answer is not a JSON object: {e}"]
        attempts.append(dict(plan=plan, raw=text if plan is None else None, errors=errors, **usage))
        if not errors:
            return plan, attempts
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "The executor rejected this plan:\n- " + "\n- ".join(errors)
                      + "\nProduce a corrected plan as JSON."}]
    raise RuntimeError("no executable plan after repairs:\n- " + "\n- ".join(attempts[-1]["errors"]))
