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
drone-only subtree, dropping the SO-ARM100), ``import_drone_usd.py`` (the
actual import + structure cleanup to a flat
``base_link``/``rotor_0..3``/``joints`` layout) and ``fix_iris650_usd.py``
(the two in-place fixes that actually made it flyable -- read that script's
docstring, it is the whole story of this asset's long "unflyable" saga).
Mass/inertia on ``base_link`` came from the import directly (MJCF's own
``<inertial>`` tags, which this bundle's own conversion notes already
derived as mass x scale^3, inertia x scale^5 from the stock Iris source)
and were cross-checked against ``iris_650.yaml``'s values -- matched to
float32 precision.

``iris_650.yaml``'s own comment used to flag ``rotor_angles``/``directions``
as unverified against the live USD. They have now been verified, directly:
with the articulation root on ``base_link`` the rotor bodies sit at
``(+-0.17107, +-0.28950 / +-0.26318, +0.03027)`` relative to ``base_link``,
whose polar angles/radii reproduce the yaml's ``rotor_angles``/
``arm_lengths`` to 4 decimal places. The yaml was right all along; what was
wrong was the frame it was being compared against.
"""

import os.path as osp

from omni_drones.robots.drone.amov_drone import AmovDrone

from aetos_multiam.assets import ASSET_PATH, USD_PATH as _ASSETS_USD_DIR
from .drone_base import DroneSpec, register_drone

MJCF_SOURCE_PATH = osp.join(ASSET_PATH, "mjcf", "iris_650_so_arm100_final", "robot.xml")
# iris_650.yaml is a small, real, hand-calibrated config (not a mesh/USD
# binary) -- kept tracked in git at assets/usd/ directly, not under the
# gitignored assets/meshes/ tree USD_PATH now points into.
PARAM_PATH = osp.join(ASSET_PATH, "usd", "iris_650.yaml")  # real, see module docstring
USD_PATH = osp.join(_ASSETS_USD_DIR, "iris_650.usd")  # real, see module docstring


class Iris650Drone(AmovDrone):
    """650mm-scaled Iris body (bare, no arm welded in), flown through
    ``AmovDrone``'s articulation machinery rather than ``MultirotorBase``'s.

    Both assets have the same shape of problem, so they get the same
    solution: a flat articulation whose ``ArticulationRootAPI`` lives on
    ``base_link``, with rotor bodies as sibling articulation links. For that
    layout ``MultirotorBase.apply_action`` is silently a no-op -- it pushes
    per-rotor thrust through ``self.rotors_view.apply_forces_and_torques_at_pos``
    (a ``RigidPrimView`` call), which PhysX ignores for bodies that belong to
    an articulation. ``AmovDrone`` already aggregates the per-rotor thrusts
    into one net force + pitch/roll torque on ``base_link`` via the
    ``ArticulationView`` force API, points its own ``ArticulationView`` at
    the ``base_link`` prim that actually carries ``ArticulationRootAPI``,
    sums body masses for the articulated total, and auto-detects arm DOFs
    (so ``hold_arm_joints``/``reset_arm_joints`` work for a runtime-mounted
    ``so_arm100`` too). Nothing in it is amov-specific except the asset
    paths and the rotor prim naming, both overridden below.

    An earlier pass carried ~150 lines of overrides here (a hand-built force
    buffer targeting ``base_link``'s body index, ``torque_axis`` taken from
    the rotors' local ``-Y``, no cosmetic rotor spin, and
    ``get_world_poses``/``get_velocities`` right-multiplying a "misalignment"
    quaternion ``(0.5,-0.5,0.5,-0.5)``). All four were chasing one single
    root cause -- the articulation root was on the ``/Iris650`` wrapper Xform
    instead of ``base_link``, so PhysX rooted the merged drone+arm
    articulation at the ARM's first body and post-reset rotated the whole
    assembly onto its edge. ``fix_iris650_usd.py`` fixes that in the asset;
    with it fixed, this asset's frames are textbook (local +Z is up for
    ``base_link`` and every rotor, spin axis == thrust axis) and all of those
    overrides are not just unnecessary but actively wrong. Deleted.
    """

    usd_path: str = USD_PATH
    param_path: str = PARAM_PATH
    hover_target_usd_path: str = USD_PATH

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        super().initialize(prim_paths_expr, track_contact_forces)
        # iris_650.yaml's mass is the BARE drone (3.4634 kg) because so_arm100
        # mounts at runtime (iris650_soarm100) -- measured articulated total is
        # 4.0954 kg. LeePositionController reads params["mass"] straight into
        # its hover feed-forward thrust and every call site builds it AFTER
        # initialize() (the ArticulationView has to exist first), so publish the
        # real total here: with the bare mass the controller under-thrusts by
        # 15% and the P-only altitude loop just settles low forever.
        self.params["mass"] = float(self.masses.reshape(-1)[0])

    def _articulation_properties_prim_path(self, spawn_root_prim_path: str) -> str:
        # Same probe as AmovDrone's (the API is on base_link for this asset
        # too, see fix_iris650_usd.py) -- only the rotor prim naming differs:
        # this asset's rotors are "rotor_0..3", amov's are "rotor_link_1..4".
        path = super()._articulation_properties_prim_path(spawn_root_prim_path)
        self._amov_rotor_glob_rel = "rotor_*"
        return path


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
