"""AerialHandover -- two heterogeneous AMs rendezvous to hand an object off.

Agents (via ``cfg/task/AerialHandover.yaml``): ``hummingbird340_dynamixel2r_bottom``
+ ``uaquad600_ambencharm_top``. Neither preset is spawnable yet (both are
MJCF-only, pending USD conversion -- see docs/ARCHITECTURE.md), so this
registers and configures correctly but cannot actually run today.

No handover object exists yet (would need a graspable item + a "transferred
from A's gripper to B's" success signal) -- this inherits
``CollaborativeAerialEnv``'s generic placeholder reward/success (converge
near a shared point) as a stand-in. Swap in real handover logic once an
object asset exists.
"""

from .collaborative_base import CollaborativeAerialEnv


class AerialHandover(CollaborativeAerialEnv):
    def _default_spawn_translation(self, index: int):
        # Close enough to plausibly exchange something, offset on either side of X.
        return (-0.75 if index == 0 else 0.75, 0.0, 1.0)

    def _goal_position(self, index: int):
        return (0.0, 0.0, 1.0)  # shared midpoint -- encourages the two agents to converge
