"""Iris scaled to a 650mm diagonal -- real, fully-specified rotor physics; USD still pending.

Unlike ``ambench_600``/``ambench_750``, this one has a complete
``RotorGroup``-compatible yaml (``assets/usd/iris_650.yaml``) -- the source
bundle (``assets/mjcf/iris_650_so_arm100_final/flight_parameters.json``)
gives exact ``force_constant``/``moment_constant``/``max_rotation_velocity``
per rotor directly, unlike AM-bench's reduced-order actuator model. What's
still missing is the USD itself: this is a *scaled* Iris (frame_scale
1.3158865458626872 from the stock ``iris.usd``'s ~0.494m wheelbase up to
0.65m), and no one has produced a scaled USD to match -- only the MJCF
meshes (``assets/mjcf/iris_650_so_arm100_final/assets/drone/``, themselves
already-scaled ``.obj`` exports). See ``assets/usd/iris_650.yaml``'s own
comments for a rotor-ordering discrepancy against stock ``iris.yaml`` that
should be checked before trusting ``directions``/``rotor_angles`` here.
"""

import os.path as osp

from omni_drones.robots.drone.iris import Iris as _Iris

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .drone_base import DroneSpec, register_drone

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "iris_650_so_arm100_final", "robot.xml")
# iris_650.yaml is a small, real, hand-calibrated config (not a mesh/USD
# binary) -- kept tracked in git at assets/usd/ directly, not under the
# gitignored assets/meshes/ tree USD_PATH now points into.
PARAM_PATH = osp.join(ASSET_PATH, "usd", "iris_650.yaml")  # real, see module docstring
USD_PATH = osp.join(_ASSETS_USD_DIR, "iris_650.usd")  # does not exist yet


class Iris650Drone(_Iris):
    """650mm-scaled Iris body (bare, no arm welded in)."""

    usd_path: str = USD_PATH
    param_path: str = PARAM_PATH
    hover_target_usd_path: str = USD_PATH


IRIS_650 = register_drone(
    DroneSpec(
        name="iris_650",
        drone_class=Iris650Drone,
        usd_path=USD_PATH,
        param_path=PARAM_PATH,
        mount_body_name="base_link",
        class_size_mm=650,
        note=(
            f"Real MJCF source at {MJCF_SOURCE_PATH}; rotor physics yaml is "
            "complete and real (assets/usd/iris_650.yaml), but there is no "
            "scaled USD body yet -- stock iris.usd is unscaled."
        ),
    )
)
