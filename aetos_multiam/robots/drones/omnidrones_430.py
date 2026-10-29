"""Firefly, 430mm class -- real asset+physics (OmniDrones), demo not wired up.

Source: ``omni_drones/robots/assets/usd/firefly.usd`` + ``firefly.yaml``
(this repo). ``arm_lengths[0] = 0.215`` in that yaml -> ~430mm rotor-to-rotor
diagonal, matching the "430mm drone" the user wants eventually. Same status
as ``omnidrones_340``: registered and usable, not exercised by a task yet.
"""

from omni_drones.robots.drone.firefly import Firefly as _Firefly

from .drone_base import DroneSpec, register_drone

OMNIDRONES_430 = register_drone(
    DroneSpec(
        name="omnidrones_430",
        drone_class=_Firefly,
        usd_path=_Firefly.usd_path,
        param_path=_Firefly.param_path,
        mount_body_name="base_link",
        class_size_mm=430,
        note="Real asset and physics; not exercised by any task yet.",
    )
)
