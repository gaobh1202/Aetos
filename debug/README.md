# debug/

Scripts that measure or verify something about an **asset or the simulator**,
not about a task or an algorithm. They exist to answer "is this robot/prop
actually what we think it is", and nothing in the benchmark imports them.

| script | answers |
|---|---|
| `measure_gripper_range.py` | how far the SO-ARM jaw opens, in millimetres, against joint angle |
| `verify_iris650_flight.py` | does `iris650_soarm100` still thrust, hover and track after an asset change |
| `record_iris650_trajectory.py` | what a scripted flight looks like, as a PNG sequence to encode |

Anything tied to a specific task stays in `scripts/` instead -- e.g.
`scripts/check_gear_assembly.py` builds the `AerialAssembly` env and asserts
*that task's* requirement (both gears seat and mesh), so it belongs with the
task runners.

Run them the same way as everything else: `conda activate omnidrone` with
`ISAACSIM_PATH` exported (see `docs/MANUAL_ZH.md` §1).
