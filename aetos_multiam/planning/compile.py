"""Plan -> execution program: named sites become poses, subtasks become reference segments for the drone policy.

The planner decides the structure; the numbers come from here. In particular a handle's ``grasp_base`` is the
nominal base pose of the arm policy's training distribution (``so101_feasible_bases.npz``:
``drone_in_handle_nominal``), i.e. where the grasp skill succeeds most often.
"""
import math
from typing import Dict, List

import numpy as np

#: Drone base relative to a handle for the grasp skill: this far outward of the bar, this far above it.
GRASP_BASE_OUT_M, GRASP_BASE_UP_M = 0.242, 0.326
#: Stand-off used to approach / leave a handle: further out and higher than the grasp base.
STANDOFF_OUT_M, STANDOFF_UP_M = 0.35, 0.15
LIFT_M = 0.25
PEAK_SPEED = {"slow": 0.3, "normal": 0.6, "fast": 1.0}  # m/s
MIN_MOVE_S, SETTLE_S, GRASP_S, RELEASE_S = 1.0, 1.0, 7.0, 2.0  # grasp: one ACT episode (175 steps at 25 Hz)


def _rz(yaw: float, v) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quintic(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class _World:
    """Where things are while the plan unfolds: carried objects move, and their handles with them."""

    def __init__(self, scene: dict):
        self.obj = {o["name"]: dict(pos=np.array(o["position"], float), yaw=math.radians(o.get("yaw_deg", 0.0)))
                    for o in scene["objects"]}
        self.part = {}
        for o in scene["objects"]:
            base = self.obj[o["name"]]
            for p in o.get("parts", []):
                self.part[p["name"]] = dict(obj=o["name"], local=_rz(-base["yaw"], np.array(p["position"], float) - base["pos"]),
                                            out_local=_rz(-base["yaw"], np.array(p["outward"], float)))
        self.home = {r["name"]: (np.array(r["position"], float), math.radians(r["yaw_deg"])) for r in scene["robots"]}

    def handle(self, name: str, obj_pose=None):
        p = self.part[name]
        o = obj_pose or self.obj[p["obj"]]
        return o["pos"] + _rz(o["yaw"], p["local"]), _rz(o["yaw"], p["out_local"])

    def base_at_handle(self, name: str, standoff: bool, obj_pose=None):
        pos, out = self.handle(name, obj_pose)
        d = GRASP_BASE_OUT_M + (STANDOFF_OUT_M if standoff else 0.0)
        up = GRASP_BASE_UP_M + (STANDOFF_UP_M if standoff else 0.0)
        return pos + out * d + np.array([0, 0, up]), math.atan2(-out[1], -out[0])  # facing the handle


def compile_plan(plan: dict, scene: dict) -> dict:
    """Execution program: one entry per subtask with its resolved reference segment and dependencies, plus a
    nominal schedule (earliest start times, assuming every skill takes its nominal time)."""
    w = _World(scene)
    pose = {r: w.home[r] for r in w.home}  # each robot's current reference pose
    last: Dict[str, str] = {}                # robot -> its previous subtask
    groups: Dict[str, List[dict]] = {}
    for s in plan["subtasks"]:
        if s["sync_group"]:
            groups.setdefault(s["sync_group"], []).append(s)
    carried_goal: Dict[str, dict] = {}       # sync_group -> the object's goal pose
    prog: Dict[str, dict] = {}
    for s in plan["subtasks"]:
        rb, site, ref = s["robot"], s["site"], s["reference"]
        p0, y0 = pose[rb]
        obj_from = obj_to = None
        if s["drone_skill"] == "hover":
            p1, y1 = p0, y0
        elif s["drone_skill"] == "carry":
            g = s["sync_group"]
            oname = w.part[s["arm_object"]]["obj"]
            cur = w.obj[oname]
            if g not in carried_goal:
                if site == "lifted":
                    goal = dict(pos=cur["pos"] + np.array([0, 0, LIFT_M]), yaw=cur["yaw"])
                else:
                    t = w.obj[ref] if ref in w.obj else dict(pos=w.handle(ref)[0], yaw=cur["yaw"])
                    goal = dict(pos=t["pos"] + np.array([0, 0, LIFT_M if site == "above" else 0.0]),
                                yaw=cur["yaw"] + _wrap(t["yaw"] - cur["yaw"]))
                carried_goal[g] = dict(obj=oname, start=dict(pos=cur["pos"].copy(), yaw=cur["yaw"]), goal=goal)
            cg = carried_goal[g]
            p1, y1 = w.base_at_handle(s["arm_object"], False, cg["goal"])
            obj_from, obj_to = cg["start"], cg["goal"]
        elif site == "home":
            p1, y1 = w.home[rb]
        else:
            p1, y1 = w.base_at_handle(ref, standoff=site != "grasp_base")
        y1 = y0 + _wrap(y1 - y0)
        if s["drone_skill"] == "hover":
            dur = {"grasped": GRASP_S, "released": RELEASE_S}.get(s["done_when"], SETTLE_S)
        else:
            path = float(np.linalg.norm(p1 - p0))
            if obj_from is not None:  # a rotating formation: the grasp point sweeps an arc
                path = max(path, abs(obj_to["yaw"] - obj_from["yaw"]) * 0.7)
            dur = max(MIN_MOVE_S, path * 1.875 / PEAK_SPEED[s["speed"]])
        deps = sorted(set(s["after"]) | ({last[rb]} if rb in last else set()))
        prog[s["id"]] = dict(
            id=s["id"], robot=rb, description=s["description"], deps=deps, sync_group=s["sync_group"],
            drone=dict(skill=s["drone_skill"], mode="station_keeping" if s["drone_skill"] == "hover" else "tracking",
                       start_pos=p0.tolist(), start_yaw=y0, goal_pos=p1.tolist(), goal_yaw=y1, duration_s=dur,
                       carried=None if obj_from is None else dict(
                           handle=s["arm_object"], obj_start_pos=obj_from["pos"].tolist(), obj_start_yaw=obj_from["yaw"],
                           obj_goal_pos=obj_to["pos"].tolist(), obj_goal_yaw=obj_to["yaw"])),
            arm=dict(skill=s["arm_skill"], object=s["arm_object"]), done_when=s["done_when"])
        pose[rb], last[rb] = (p1, y1), s["id"]
        # The object arrives when the last member of its carry group has been compiled.
        if s["drone_skill"] == "carry" and s is groups[s["sync_group"]][-1]:
            w.obj[carried_goal[s["sync_group"]]["obj"]] = carried_goal[s["sync_group"]]["goal"]
    # Members of a sync group wait for each other's dependencies and take the slowest member's time.
    for g, members in groups.items():
        deps = sorted({d for m in members for d in prog[m["id"]]["deps"]} - {m["id"] for m in members})
        dur = max(prog[m["id"]]["drone"]["duration_s"] for m in members)
        for m in members:
            prog[m["id"]]["deps"], prog[m["id"]]["drone"]["duration_s"] = deps, dur
    # Nominal schedule.
    for sid in [s["id"] for s in plan["subtasks"]]:
        e = prog[sid]
        e["t_start"] = max([prog[d]["t_end"] for d in e["deps"]], default=0.0)
        e["t_end"] = e["t_start"] + e["drone"]["duration_s"]
    for g, members in groups.items():  # a later member may have pushed the group's start
        t0 = max(prog[m["id"]]["t_start"] for m in members)
        for m in members:
            prog[m["id"]]["t_start"], prog[m["id"]]["t_end"] = t0, t0 + prog[m["id"]]["drone"]["duration_s"]
    return dict(robots=sorted(w.home), subtasks=[prog[s["id"]] for s in plan["subtasks"]],
                duration_s=max((e["t_end"] for e in prog.values()), default=0.0),
                home={r: dict(pos=w.home[r][0].tolist(), yaw=w.home[r][1]) for r in w.home},
                handles={n: dict(obj=p["obj"], local=p["local"].tolist(), out_local=p["out_local"].tolist())
                         for n, p in w.part.items()})


def segment_ref(program: dict, e: dict, tau: float):
    """(position (3,), yaw) of subtask e's reference, tau seconds after it started (held at the goal afterwards)."""
    d = e["drone"]
    u = float(quintic(tau / max(d["duration_s"], 1e-6)))
    c = d["carried"]
    if c is not None:  # formation: interpolate the OBJECT's pose, then place the base at its handle
        h = program["handles"][c["handle"]]
        pos = np.array(c["obj_start_pos"]) + u * (np.array(c["obj_goal_pos"]) - np.array(c["obj_start_pos"]))
        yaw = c["obj_start_yaw"] + u * (c["obj_goal_yaw"] - c["obj_start_yaw"])
        out = _rz(yaw, h["out_local"])
        p = pos + _rz(yaw, h["local"]) + out * GRASP_BASE_OUT_M + np.array([0, 0, GRASP_BASE_UP_M])
        return p, d["start_yaw"] + _wrap(math.atan2(-out[1], -out[0]) - d["start_yaw"])
    return (np.array(d["start_pos"]) + u * (np.array(d["goal_pos"]) - np.array(d["start_pos"])),
            d["start_yaw"] + u * (d["goal_yaw"] - d["start_yaw"]))


def reference_at(program: dict, robot: str, t: float):
    """(position (3,), yaw, mode, subtask id) of a robot's reference on the nominal schedule."""
    cur = None
    for e in program["subtasks"]:
        if e["robot"] == robot and t >= e["t_start"]:
            cur = e
    if cur is None:
        h = program["home"][robot]
        return np.array(h["pos"]), h["yaw"], "station_keeping", None
    pos, yaw = segment_ref(program, cur, t - cur["t_start"])
    return pos, yaw, cur["drone"]["mode"] if t < cur["t_end"] else "station_keeping", cur["id"]
