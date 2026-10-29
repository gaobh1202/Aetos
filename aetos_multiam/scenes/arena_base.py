"""Arena base -- ground plane + lighting for the template env (robosuite ``Arena`` analog).

The env-origin grid itself is handled by ``IsaacEnv``'s ``GridCloner`` already
(inherited from OmniDrones); an ``Arena`` here only populates what's inside
one template env before cloning, same division of labor as this repo's own
``_design_scene`` implementations (e.g. ``hover.py``'s ground-plane call).
"""

from typing import List

import omni_drones.utils.kit as kit_utils


class Arena:
    """Base arena: just a ground plane. Subclass to add walls/fixtures."""

    ground_plane_prim_path = "/World/defaultGroundPlane"

    def spawn(self, template_env_ns: str) -> List[str]:
        """Populate the template env; return prim paths needing global collision filtering."""
        kit_utils.create_ground_plane(
            self.ground_plane_prim_path,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        return [self.ground_plane_prim_path]
