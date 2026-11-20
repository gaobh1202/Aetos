# OmniDrones changes Aetos depends on

Aetos imports `omni_drones` from a fork: https://github.com/gaobh1202/amov_omni
at the commit in `BASE_COMMIT`, plus local changes that are not on that fork yet:

- `aetos.patch`: edits to tracked files (shared physics view for several robots in one
  scene, drones without rotor rigid bodies, the thrust + body-rate action transform, ...).
- `overlay/`: files that do not exist on the fork (controller gains for the Aetos drones,
  `amov_drone_arm_1.yaml`, the arm-trajectory / shelf-grasp envs and their configs, scripts).

Set up:

    git clone https://github.com/gaobh1202/amov_omni.git OmniDrones
    cd OmniDrones && git checkout "$(cat /path/to/Aetos_MultiAM/third_party/omnidrones/BASE_COMMIT)"
    bash /path/to/Aetos_MultiAM/third_party/omnidrones/apply.sh .
    pip install -e .

## Isaac Sim 6 port (branch `isaacsim6`)

On this branch `aetos.patch` and `overlay/` also carry the port of OmniDrones itself
from Isaac Sim 4.1 to 6.0 (Python 3.12, torch 2.11, torchrl 0.15) -- `apply.sh` on a
checkout at `BASE_COMMIT` gives a tree that runs on Isaac Sim 6. What the port had to
change, in both OmniDrones and `aetos_multiam`:

| 4.1 | 6.0 |
|---|---|
| `omni.isaac.core.*` | `isaacsim.core.api.*` (`objects`, `materials`, `simulation_context`, `world`), `isaacsim.core.utils.*`, `isaacsim.core.prims` |
| `RigidPrimView` / `XFormPrimView` / `ArticulationView` | `isaacsim.core.prims.RigidPrim` / `XFormPrim` / `Articulation` (the single-prim classes became `Single*`) -- imported under their old local names |
| `omni.isaac.cloner`, `.debug_draw`, `.kit`, `.version`, `.sensor` | `isaacsim.core.cloner`, `isaacsim.util.debug_draw`, `isaacsim`, `isaacsim.core.version`, `isaacsim.sensors.camera` |
| `omni.isaac.core.utils.nucleus` | `isaacsim.storage.native` |
| `omni.isaac.dynamic_control` | removed -- forces go through a `RigidPrim` tensor view, prim classification through USD schemas |
| `$EXP_PATH/omni.isaac.sim.python.kit` | none; `SimulationApp(config)` picks `isaacsim.exp.base.python.kit` |
| `SimulationContext._physics_sim_view`, per-view `_physics_sim_view` | `SimulationContext.physics_sim_view` / `SimulationManager.get_physics_sim_view()`; views build their tensor view in the base class' `_on_physics_ready` |
| sim param `use_flatcache` | `use_fabric` (mapped in `IsaacEnv`) |
| `omni.replicator.isaac`, `omni.kit.viewport.pxr` extensions | gone / not bundled -- no longer enabled |
| `PhysxMaterialAPI.improvePatchFriction` | removed from the schema |
| torchrl `CompositeSpec`, `UnboundedContinuousTensorSpec`, `BoundedTensorSpec`, `DiscreteTensorSpec`, ... | `Composite`, `Unbounded`, `Bounded`, `Categorical`, ... |
| torchrl `SyncDataCollector` | `Collector` |
| `env.action_spec` / `env.reward_spec` unwrap to the leaf for any single key | only for a key literally named `action` / `reward` -- `IsaacEnv` restores the old behaviour |
| `tensordict.nn.make_functional` + `vmap(module)(x, params)` | removed -- `RotorGroup` takes an explicit params TensorDict, MAPPO uses `TensorDict.from_module` / `to_module` |
| dataclass fields defaulting to a config instance | `field(default_factory=...)` (Python 3.11+ rejects the former) |

Not ported: `mappo_old`, `happo` and `matd3` (still written against `make_functional`), and the
`Forest` / `Pinball` envs (need Isaac Lab). All of them are skipped at import instead of
taking the package down. `HoverShelfGrasp` imports a module from the raids-am-simulator
checkout and is skipped the same way when that checkout is absent.
