"""Iris scaled to a 650mm diagonal -- real, fully-specified rotor physics AND a real USD.

Unlike ``ambench_600``/``ambench_750``, this one has a complete
``RotorGroup``-compatible yaml (``assets/usd/iris_650.yaml``) -- the source
bundle (``assets/mjcf/iris_650_so_arm100_final/flight_parameters.json``)
gives exact ``force_constant``/``moment_constant``/``max_rotation_velocity``
per rotor directly, unlike AM-bench's reduced-order actuator model.

The USD (``assets/meshes/usd/iris_650.usd``) is built by importing this
same bundle's own already-scaled geometry (frame_scale 1.3158865458626872
from the stock ``iris.usd``'s ~0.494m wheelbase up to 0.65m) via
``omni.importer.mjcf`` -- see
``assets/mjcf/iris_650_so_arm100_final/split_drone.py`` (extracts the
drone-only subtree, dropping the SO-ARM100) and ``import_drone_usd.py``
(the actual import + structure cleanup: renames/promotes the imported body
to a flat ``base_link``/``rotor_0..3``/``joints`` layout matching every
other drone in this project, puts ``ArticulationRootAPI`` on the file's own
top-level prim rather than ``base_link`` -- required for
``RobotBase.initialize()`` to find it at the plain spawn-root prim path --
and converts the 4 rotor attachments from ``PhysicsFixedJoint`` to
``PhysicsRevoluteJoint`` (axis Z, no limits): the source MJCF deliberately
gives rotors no spin DOF, but Isaac's own ``ArticulationView.post_reset()``
crashes on a literally zero-DOF articulation (``_default_joints_state`` is
``None``) -- the revolute conversion is a required workaround, not a
cosmetic choice, confirmed empirically by reproducing the crash with the
original ``PhysicsFixedJoint`` version first). Mass/inertia on ``base_link``
came from the import directly (MJCF's own ``<inertial>`` tags, which this
bundle's own conversion notes already derived as mass x scale^3, inertia x
scale^5 from the stock Iris source) and were cross-checked against
``iris_650.yaml``'s values -- matched to float32 precision.

Verified end-to-end via the real ``Iris650Drone``/``MultirotorBase`` class
(not just raw USD inspection): spawns cleanly, ``num_rotors=4``, mass/inertia
read back match the yaml, and holds its spawn pose with zero drift over 1s
of simulated time (gravity zeroed, no thrust applied) -- see
``scripts/verify_iris650_bare.py``.

See ``assets/usd/iris_650.yaml``'s own comments for a rotor-ordering
discrepancy against stock ``iris.yaml`` that should be checked before
trusting ``directions``/``rotor_angles`` here -- not re-investigated as part
of building the USD, since ``RotorGroup`` only reads this yaml, not body
geometry, for that.
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
USD_PATH = osp.join(_ASSETS_USD_DIR, "iris_650.usd")  # real, see module docstring


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
            "complete and real (assets/usd/iris_650.yaml); scaled USD body "
            f"imported from the same bundle's geometry (see {USD_PATH} and "
            "this module's own docstring for the import/cleanup pipeline)."
        ),
    )
)
