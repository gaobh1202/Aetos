"""AerialAssemblyTriple -- three identical AMs jointly assemble a part.

Agents (via ``cfg/task/AerialAssemblyTriple.yaml``): three
``amov950_airbot_bottom`` -- the one spawnable preset, so (like
``AerialTransport``) this is one of the two collaborative tasks that can
genuinely be smoke-tested end-to-end today, not just registered.

No assembly fixture/parts asset exists yet -- this inherits
``CollaborativeAerialEnv``'s generic placeholder reward/success (converge
near a shared point) as a stand-in for "three parts brought together".
"""

import math

from .collaborative_base import CollaborativeAerialEnv


class AerialAssemblyTriple(CollaborativeAerialEnv):
    def _default_spawn_translation(self, index: int):
        angle = 2 * math.pi * index / 3
        radius = 1.0
        return (radius * math.cos(angle), radius * math.sin(angle), 1.2)

    def _goal_position(self, index: int):
        return (0.0, 0.0, 1.2)  # shared assembly point at the circle's center
