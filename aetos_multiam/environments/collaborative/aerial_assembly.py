"""AerialAssembly -- two heterogeneous AMs converge to jointly assemble a part.

Agents (via ``cfg/task/AerialAssembly.yaml``): ``uaquad600_ambencharm_top``
+ ``amov950_airbot_bottom``. Only the second preset is spawnable today
(the first is MJCF-only, pending USD conversion), so this registers and
configures correctly but cannot actually run as a two-AM task yet.

No assembly fixture/parts asset exists yet -- this inherits
``CollaborativeAerialEnv``'s generic placeholder reward/success (converge
near a shared point) as a stand-in for "bring the two parts together".
"""

from .collaborative_base import CollaborativeAerialEnv


class AerialAssembly(CollaborativeAerialEnv):
    def _default_spawn_translation(self, index: int):
        return (-0.75 if index == 0 else 0.75, 0.0, 1.2)

    def _goal_position(self, index: int):
        return (0.0, 0.0, 1.2)  # shared assembly point
