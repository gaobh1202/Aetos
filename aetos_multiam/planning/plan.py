"""The skill library, the plan format a planner must produce, and its validation.

A plan does not contain numbers. It names WHO does WHAT with WHICH skill, relative to WHICH object,
in what order; ``compile.py`` turns the named sites into poses and trajectories.

Scene (input), a dict::

    {"instruction": str,
     "robots":  [{"name": "am_0", "position": [x, y, z], "yaw_deg": float}, ...],
     "objects": [{"name": "carry_box", "position": [...], "yaw_deg": float, "mass_kg": float,
                  "parts": [{"name": "handle_0", "position": [...], "outward": [x, y, z]}, ...]},
                 {"name": "drop_zone", "position": [...], "yaw_deg": float}, ...]}

Plan (output): see PLAN_SCHEMA.
"""
from typing import Dict, List

#: What the drone's RL policy is asked to do during a subtask.
DRONE_SKILLS = {
    "goto": "Fly to a site. The policy tracks a reference point that MOVES along a smooth path (tracking mode).",
    "hover": "Hold the site reached by the previous subtask. The reference point is FIXED (station-keeping mode); "
             "use it whenever the arm manipulates, so the arm works from a steady base.",
    "carry": "Move a jointly held object to a site, in formation: every robot holding it flies the path its own grasp "
             "point follows when the object moves rigidly. All holders must be in the same sync_group.",
}
#: What the arm does during a subtask.
ARM_SKILLS = {
    "stow": "Arm folded in its default pose (flight between places).",
    "grasp": "Learned visuomotor policy (ACT): reach the named handle and close the gripper on it. Needs the drone "
             "hovering at that handle's 'grasp_base' site; its success drops quickly outside that region.",
    "hold": "Keep the gripper closed on what it holds.",
    "release": "Open the gripper and withdraw from the handle.",
}
#: Named places a drone can be sent to, relative to `reference` (an object or one of its parts).
SITES = {
    "pregrasp_base": "Stand-off pose behind and above a handle, facing it. Approach a handle through this site.",
    "grasp_base": "Hover pose from which the arm's grasp skill reaches a handle (reference must be a handle).",
    "lifted": "carry only: the held object raised straight up, clear of what it rests on.",
    "above": "carry only: the held object centred over the reference, still raised.",
    "on": "carry only: the held object set down on the reference, aligned with it.",
    "retreat": "Back away from a handle after releasing it, to a safe distance.",
    "home": "The robot's own start pose (reference: the robot's own name).",
}
DONE_WHEN = {
    "arrived": "the drone is at the site and steady",
    "grasped": "the gripper holds the handle",
    "released": "the gripper is open and clear of the handle",
}
SPEEDS = ("slow", "normal", "fast")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Brief: how the task was decomposed and why."},
        "allocation": {
            "type": "array",
            "description": "Which robot takes which object part, and why.",
            "items": {
                "type": "object",
                "properties": {"robot": {"type": "string"}, "part": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["robot", "part", "reason"], "additionalProperties": False,
            },
        },
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "robot": {"type": "string"},
                    "description": {"type": "string"},
                    "drone_skill": {"type": "string", "enum": list(DRONE_SKILLS)},
                    "site": {"type": "string", "enum": list(SITES)},
                    "reference": {"type": "string", "description": "Object or part name the site is relative to."},
                    "speed": {"type": "string", "enum": list(SPEEDS)},
                    "arm_skill": {"type": "string", "enum": list(ARM_SKILLS)},
                    "arm_object": {"type": "string", "description": "Part the arm acts on; empty string if none."},
                    "done_when": {"type": "string", "enum": list(DONE_WHEN)},
                    "after": {"type": "array", "items": {"type": "string"},
                              "description": "Ids of subtasks (any robot) that must be done before this starts."},
                    "sync_group": {"type": "string", "description": "Subtasks sharing a non-empty group start "
                                   "together and end together (joint motion). Empty string if none."},
                },
                "required": ["id", "robot", "description", "drone_skill", "site", "reference", "speed", "arm_skill",
                             "arm_object", "done_when", "after", "sync_group"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "allocation", "subtasks"],
    "additionalProperties": False,
}


def _names(scene: dict):
    robots = [r["name"] for r in scene["robots"]]
    objects = {o["name"]: o for o in scene["objects"]}
    parts = {p["name"]: o["name"] for o in scene["objects"] for p in o.get("parts", [])}
    return robots, objects, parts


def validate_plan(plan: dict, scene: dict) -> List[str]:
    """Problems that make the plan unexecutable (empty list: executable). Messages are written to be handed
    back to the planner."""
    errs: List[str] = []
    robots, objects, parts = _names(scene)
    subtasks = plan.get("subtasks", [])
    ids = [s["id"] for s in subtasks]
    if len(set(ids)) != len(ids):
        errs.append("subtask ids are not unique")
    seen: set = set()
    holding: Dict[str, str] = {}   # robot -> part it holds
    held_by: Dict[str, str] = {}   # part -> robot
    at: Dict[str, tuple] = {}      # robot -> (site, reference) of its last drone target
    groups: Dict[str, List[dict]] = {}
    for s in subtasks:
        tag = f"subtask {s['id']}"
        if s["robot"] not in robots:
            errs.append(f"{tag}: unknown robot {s['robot']!r} (robots: {robots})")
            continue
        for a in s["after"]:
            if a not in seen:
                errs.append(f"{tag}: 'after' names {a!r}, which is not an earlier subtask (list subtasks in a valid order)")
        ref, site, rb = s["reference"], s["site"], s["robot"]
        if site == "home":
            if ref != rb:
                errs.append(f"{tag}: site 'home' needs reference = the robot's own name")
        elif site in ("pregrasp_base", "grasp_base", "retreat"):
            if ref not in parts:
                errs.append(f"{tag}: site {site!r} needs a handle as reference, got {ref!r} (parts: {list(parts)})")
        elif ref not in objects and ref not in parts:
            errs.append(f"{tag}: unknown reference {ref!r}")
        if s["drone_skill"] == "hover" and at.get(rb) != (site, ref):
            errs.append(f"{tag}: hover must name the site the robot already reached ({at.get(rb)}), not ({site}, {ref})")
        if s["drone_skill"] == "carry":
            if site not in ("lifted", "above", "on"):
                errs.append(f"{tag}: carry takes site lifted/above/on, not {site!r}")
            if rb not in holding:
                errs.append(f"{tag}: carry, but {rb} holds nothing at this point")
            if not s["sync_group"]:
                errs.append(f"{tag}: carry needs a sync_group shared by every robot holding the object")
        elif site in ("lifted", "above", "on") and s["drone_skill"] != "hover":
            errs.append(f"{tag}: site {site!r} is reached with the carry skill, not {s['drone_skill']}")
        elif rb in holding and s["drone_skill"] == "goto":
            errs.append(f"{tag}: {rb} holds {holding[rb]} and may only hover or carry until it releases")
        arm, obj = s["arm_skill"], s["arm_object"]
        if arm == "grasp":
            if obj not in parts:
                errs.append(f"{tag}: grasp needs arm_object = a handle (parts: {list(parts)})")
            elif held_by.get(obj) not in (None, rb):
                errs.append(f"{tag}: {obj} is already held by {held_by[obj]}")
            if s["drone_skill"] != "hover" or (site, ref) != ("grasp_base", obj):
                errs.append(f"{tag}: grasp needs the drone hovering at grasp_base of {obj}")
            if s["done_when"] != "grasped":
                errs.append(f"{tag}: a grasp subtask ends on done_when = grasped")
            if rb in holding:
                errs.append(f"{tag}: {rb} already holds {holding[rb]}")
            holding[rb], held_by[obj] = obj, rb
        elif arm == "hold":
            if rb not in holding:
                errs.append(f"{tag}: hold, but {rb} holds nothing")
        elif arm == "release":
            if rb not in holding:
                errs.append(f"{tag}: release, but {rb} holds nothing")
            else:
                held_by.pop(holding.pop(rb), None)
            if s["drone_skill"] != "hover":
                errs.append(f"{tag}: release needs the drone hovering")
            if s["done_when"] != "released":
                errs.append(f"{tag}: a release subtask ends on done_when = released")
        elif arm == "stow" and rb in holding:
            errs.append(f"{tag}: stow while holding {holding[rb]}; use hold")
        if arm not in ("grasp", "release") and s["done_when"] != "arrived":
            errs.append(f"{tag}: done_when {s['done_when']!r} needs the matching arm skill")
        if s["sync_group"]:
            groups.setdefault(s["sync_group"], []).append(s)
        at[rb] = (site, ref)
        seen.add(s["id"])
    for g, members in groups.items():
        if len({m["robot"] for m in members}) != len(members):
            errs.append(f"sync_group {g!r}: one subtask per robot")
        if len({(m["drone_skill"], m["site"], m["done_when"]) for m in members}) != 1:
            errs.append(f"sync_group {g!r}: members must share drone_skill, site and done_when")
        if members[0]["drone_skill"] == "carry":
            obj = {parts.get(m["arm_object"]) for m in members} - {None}
            if len(obj) != 1:
                errs.append(f"sync_group {g!r}: carry members must hold (arm_object) handles of one object")
            else:
                need = {p for p, o in parts.items() if o == next(iter(obj))}
                have = {m["arm_object"] for m in members}
                # Everything that holds the object has to move with it.
                if not have <= need or len(have) != len(members):
                    errs.append(f"sync_group {g!r}: each member's arm_object must be the handle it holds")
    if holding:
        errs.append(f"the plan ends with {holding} still held; release it")
    return errs
