"""AerialTransport -- two identical AMs cooperatively transport a payload to a destination.

Agents (via ``cfg/task/AerialTransport.yaml``): two ``amov950_airbot_bottom``
-- the one preset that's actually spawnable today, so this is the first of
the five multi-AM tasks that can genuinely be smoke-tested end-to-end (run
``scripts/run_button_grasp.py``-style, once a matching launcher script
exists for this task).

No shared-payload asset or joint-attachment logic exists yet (a true
"both drones rigidly carry one object" setup needs either a closed-loop
dual fixed-joint, which is its own piece of engineering, or a simpler
two-independent-payloads approximation) -- this inherits
``CollaborativeAerialEnv``'s generic placeholder reward/success (both
agents reach a shared destination region) as a stand-in for "transported".
"""

from .collaborative_base import CollaborativeAerialEnv


class AerialTransport(CollaborativeAerialEnv):
    def _default_spawn_translation(self, index: int):
        return (0.0, 1.0 if index == 0 else -1.0, 1.0)

    def _goal_position(self, index: int):
        x, y, _ = self._default_spawn_translation(index)
        return (x + 4.0, y, 1.5)  # fly the formation 4m forward and up -- the "destination"
