"""Asset path resolution shared by every drone/manipulator/gripper/object spec.

Mirrors robosuite's ``xml_path_completion`` / omni_drones's ``ASSET_PATH``: one
place that knows where USD files live, plus a clear error for specs whose
asset hasn't been dropped in yet (see ``docs/ARCHITECTURE.md``'s asset table).

``MESHES_PATH``/``USD_PATH`` point into ``assets/meshes/`` -- the one folder
holding every binary model file (USD + every MJCF bundle's mesh geometry),
distributed via netdisk rather than committed to git (see ``docs/ASSETS.md``).
Every manipulator/gripper spec in this project derives its ``usd_path`` from
``USD_PATH`` (not a hardcoded ``assets/usd/...`` string), so this one
constant is the single place that knows where the folder actually is.
"""

import os.path as osp

ASSET_PATH = osp.join(osp.dirname(__file__), "assets")
MESHES_PATH = osp.join(ASSET_PATH, "meshes")
USD_PATH = osp.join(MESHES_PATH, "usd")


class AssetNotAvailableError(RuntimeError):
    """Raised when a spec is instantiated for spawning but has no real asset yet.

    This is a scaffolding marker, not a bug: the spec class, its registry
    entry, and its physical parameters (where known) all exist already --
    only the USD/MJCF file is missing. Drop the file at ``expected_path`` (or
    point the spec at wherever it actually lives) and this error goes away.
    """

    def __init__(self, component_name: str, expected_path: str, note: str = ""):
        message = (
            f"'{component_name}' has no asset at '{expected_path}' yet. Either "
            "this component is a registered placeholder with no real asset at "
            "all (see docs/ARCHITECTURE.md's asset table), or the real USD file "
            "just hasn't been downloaded -- model binaries aren't committed to "
            "git, see docs/ASSETS.md to fetch them."
        )
        if note:
            message += f" {note}"
        super().__init__(message)
        self.component_name = component_name
        self.expected_path = expected_path
