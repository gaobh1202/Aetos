"""Arena base -- ground plane + lighting for the template env (robosuite ``Arena`` analog).

The env-origin grid itself is handled by ``IsaacEnv``'s ``GridCloner`` already
(inherited from OmniDrones); an ``Arena`` here only populates what's inside
one template env before cloning, same division of labor as this repo's own
``_design_scene`` implementations (e.g. ``hover.py``'s ground-plane call).
"""

from typing import List

import omni_drones.utils.kit as kit_utils


class Arena:
    """Base arena -- the default scene: ground plane, nothing else.

    Subclass to add walls/fixtures. Usable on its own as the empty scene
    every other one starts from.
    """

    ground_plane_prim_path = "/World/defaultGroundPlane"
    ground_color = (1.0, 1.0, 1.0)
    """Tint on Isaac's own ``Environments/Grid/default_environment.usd``.

    White, because that is what AM-Bench's ground looks like -- its
    ``spawn_ground_plane(cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0)))`` loads
    the same USD and tints it the same way (``ambench/tasks/base/base_env.py``).
    OmniDrones' own default for the same call is a near-black
    ``(0.065, 0.0725, 0.080)``, which is what the mission videos were shot on.
    """

    def spawn(self, template_env_ns: str) -> List[str]:
        """Populate the template env; return prim paths needing global collision filtering."""
        kit_utils.create_ground_plane(
            self.ground_plane_prim_path,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            color=self.ground_color,
        )
        return [self.ground_plane_prim_path]
