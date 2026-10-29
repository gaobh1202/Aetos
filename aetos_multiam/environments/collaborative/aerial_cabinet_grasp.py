"""AerialCabinetGrasp -- three AMs (two hexacopter-arm + one quad-arm) around a cabinet.

Agents (via ``cfg/task/AerialCabinetGrasp.yaml``): two
``fahexa750_ambencharm_top`` + one ``iris650_soarm100_bottom``. None of the
three presets are spawnable yet (all MJCF-only, pending USD conversion), so
this registers and configures correctly but cannot actually run today.

No cabinet asset exists yet (AM-bench's own ``cabinet_pick_place`` task has
one, but it's an IsaacLab-side asset not ported here) -- this inherits
``CollaborativeAerialEnv``'s generic placeholder reward/success (converge
near a shared point, standing in for "cabinet location") as a stand-in.
"""

from .collaborative_base import CollaborativeAerialEnv


class AerialCabinetGrasp(CollaborativeAerialEnv):
    def _default_spawn_translation(self, index: int):
        # Three agents fanned out in front of the (not-yet-modeled) cabinet.
        offsets = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.2)]
        dx, dy = offsets[index % len(offsets)]
        return (dx, dy, 1.3)

    def _goal_position(self, index: int):
        return (0.0, 1.5, 1.3)  # the (placeholder) cabinet location
