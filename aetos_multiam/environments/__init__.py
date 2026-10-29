"""The one place every task gets listed -- robosuite ``environments/base.py``'s
``REGISTERED_ENVS``/``make()`` pattern, answering "通过 environment 模块注册环境类"."""

from .base_env import AetosEnv
from .aerial_manipulation_env import AerialManipulationEnv

# Import side effects populate AetosEnv.REGISTERED_ENVS (dexmimicgen-style:
# importing a task module registers its class via __init_subclass__).
from .single import button_grasp  # noqa: F401
from . import collaborative  # noqa: F401

REGISTERED_ENVS = AetosEnv.REGISTERED_ENVS


def make(env_name: str, cfg, headless: bool) -> AetosEnv:
    if env_name not in REGISTERED_ENVS:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {sorted(REGISTERED_ENVS)}"
        )
    return REGISTERED_ENVS[env_name](cfg, headless)


__all__ = ["AetosEnv", "AerialManipulationEnv", "REGISTERED_ENVS", "make"]
