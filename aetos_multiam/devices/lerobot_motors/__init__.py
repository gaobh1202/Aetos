"""Vendored Feetech motor-bus driver (serial protocol only, zero sim coupling).

Copied verbatim (Apache-2.0, headers preserved) from
https://github.com/LightwheelAI/leisaac -- itself a trimmed copy of
HuggingFace lerobot's own ``lerobot/common/motors`` module
(https://github.com/huggingface/lerobot). Only two external deps:
``pyserial`` and ``deepdiff`` -- nothing here imports IsaacLab or Isaac Sim,
so it's safe to vendor directly instead of pulling in the full ``lerobot``
pip package (which drags in its own torch/training stack) just to talk to
an SO101 leader arm's Feetech STS3215 servos over USB.

Not modified beyond this docstring and directory flattening (``common/`` ->
``lerobot_motors/``, same internal nesting so the relative imports
(``..errors``, ``..utils``, ``..motors_bus``) still resolve unchanged).
"""

from .motors import FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode, OperatingMode

__all__ = ["FeetechMotorsBus", "Motor", "MotorCalibration", "MotorNormMode", "OperatingMode"]
