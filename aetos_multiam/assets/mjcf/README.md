# MJCF source bundles -- not directly usable by Isaac Sim

These four directories are complete, self-contained MuJoCo ("独立运行包")
exports of aerial-manipulator combinations, each with its own `robot.xml`,
meshes, a `README.md`, and (where noted) a `CONVERSION_NOTES.md` /
`conversion_report.json` documenting exactly where every number came from.
They are **viewer/kinematics exports** of the original Isaac Sim / USD
assemblies -- explicitly *not* ported controllers, aerodynamics, or task
environments (see each bundle's own `CONVERSION_NOTES.md`/`README.md`).

Isaac Sim (`omni.isaac.core`) needs USD, not MJCF, so none of these can be
spawned by `aetos_multiam` as-is. They're kept here as the **authoritative
source of truth** for joint names, mount transforms, mass/inertia, and
(where given) rotor allocation constants -- the registered specs in
`aetos_multiam/robots/{drones,manipulators,grippers}/` that reference them
cite exactly which numbers came from which file. Converting a bundle to USD
(e.g. via Isaac Sim's MJCF importer) and pointing the matching spec's
`usd_path` at the result is the remaining step to make it spawnable.

| Directory | Combination | Component specs | Preset name |
|---|---|---|---|
| `ambench_quad_600_top_arm_final` | AM-bench UAQuad 600mm, arm top-mounted | `ambench_600` drone + `ambench_quad_arm` manipulator | `uaquad600_ambencharm_top` |
| `ambench_fa_hexa_750_final` | AM-bench FAHexa 750mm, tilt-rotor hexacopter | `ambench_750` drone + `ambench_hexa_arm` manipulator | `fahexa750_ambencharm_top` |
| `hummingbird_2r_three_finger_50pct` | OmniDrones Hummingbird 340mm + 50%-scale dynamixel 2R + three-finger gripper | `omnidrones_340` drone + `dynamixel_2r` manipulator + `three_finger_softgripper` gripper | `hummingbird340_dynamixel2r_bottom` |
| `iris_650_so_arm100_final` | OmniDrones Iris scaled to 650mm + full-size SO-ARM100 | `iris_650` drone + `so_arm100` manipulator (gripper bundled) | `iris650_soarm100_bottom` |

See `robots/presets.py` for all five ready-made `AerialManipulatorCfg`s
(including the pre-existing `amov950_airbot_bottom` combo) this maps to, and
`cfg/task/Aerial*.yaml` for the multi-agent tasks that reference them by name.
