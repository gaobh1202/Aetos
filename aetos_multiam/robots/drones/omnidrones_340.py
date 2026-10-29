"""Hummingbird, 340mm class -- real asset+physics (OmniDrones), demo not wired up.

Source: ``omni_drones/robots/assets/usd/hummingbird.usd`` +
``hummingbird.yaml`` (this repo). ``arm_lengths[0] = 0.17`` in that yaml ->
~340mm rotor-to-rotor diagonal, matching the "340mm drone" the user wants
eventually. Registered with its real, working OmniDrones class/asset so
it's ready to use; no task exercises it in this pass (only ``amov_950``
does, via ``ButtonGrasp``).
"""

from omni_drones.robots.drone.hummingbird import Hummingbird as _Hummingbird

from .drone_base import DroneSpec, register_drone

OMNIDRONES_340 = register_drone(
    DroneSpec(
        name="omnidrones_340",
        drone_class=_Hummingbird,
        usd_path=_Hummingbird.usd_path,
        param_path=_Hummingbird.param_path,
        mount_body_name="base_link",
        class_size_mm=340,
        note="Real asset and physics; not exercised by any task yet.",
    )
)
