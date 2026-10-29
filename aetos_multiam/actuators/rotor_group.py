"""Rotor thrust/torque allocation -- re-exported from OmniDrones.

See the "Dependency on OmniDrones" note in docs/ARCHITECTURE.md: this is
generic, physically-grounded drone actuator code, not benchmark logic, so
it's imported rather than duplicated.
"""

from omni_drones.actuators.rotor_group import RotorGroup

__all__ = ["RotorGroup"]
