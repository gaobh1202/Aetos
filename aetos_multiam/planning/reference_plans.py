"""Hand-written plans in the planner's own format: what the scripted demos do, for comparison with the VLM's."""


def transport_plan(robot_handle=(("am_0", "handle_0"), ("am_1", "handle_1")), box="carry_box", target="drop_zone") -> dict:
    def st(i, rb, desc, drone, site, ref, speed, arm, obj, done, after=(), group=""):
        return dict(id=i, robot=rb, description=desc, drone_skill=drone, site=site, reference=ref, speed=speed,
                    arm_skill=arm, arm_object=obj, done_when=done, after=list(after), sync_group=group)

    sub, grasps = [], []
    for k, (rb, h) in enumerate(robot_handle):
        sub += [st(f"approach_{k}", rb, f"fly to the stand-off of {h}", "goto", "pregrasp_base", h, "fast", "stow", "", "arrived"),
                st(f"close_in_{k}", rb, f"move in to the grasp base of {h}", "goto", "grasp_base", h, "slow", "stow", "", "arrived"),
                st(f"grasp_{k}", rb, f"hold still while the arm grasps {h}", "hover", "grasp_base", h, "slow", "grasp", h, "grasped")]
        grasps.append(f"grasp_{k}")
    for name, site, ref, prev in (("lift", "lifted", box, grasps), ("transport", "above", target, ["lift_0"]),
                                  ("lower", "on", target, ["transport_0"])):
        sub += [st(f"{name}_{k}", rb, f"{name} the box together", "carry", site, ref, "slow", "hold", h, "arrived",
                   prev, name) for k, (rb, h) in enumerate(robot_handle)]
    for k, (rb, h) in enumerate(robot_handle):
        sub += [st(f"release_{k}", rb, f"let go of {h}", "hover", "on", target, "slow", "release", h, "released",
                   [f"lower_{j}" for j in range(len(robot_handle))]),
                st(f"retreat_{k}", rb, f"back away from {h}", "goto", "retreat", h, "slow", "stow", "", "arrived"),
                st(f"home_{k}", rb, "return to the start zone", "goto", "home", rb, "fast", "stow", "", "arrived")]
    return dict(reasoning="Each robot takes the handle on its own side; both grasp, then carry in formation.",
                allocation=[dict(robot=rb, part=h, reason="nearest handle") for rb, h in robot_handle], subtasks=sub)
