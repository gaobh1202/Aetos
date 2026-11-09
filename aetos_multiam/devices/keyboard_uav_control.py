"""Keyboard-polled UAV flight-stick input device.

Reads a normalized flight-stick intent straight off the keyboard each frame
via carb's POLLING API (``carb.input.IInput.get_keyboard_value``, confirmed
available in this Isaac Sim install) -- no event-callback registration, same
choice ``soarm_leader.py`` made (see that module's own docstring: this
project doesn't use IsaacLab's callback-based ``Device``/``DeviceBase``).
``examples/aerial_manipulator_control_demo.py`` drives its own
``while simulation_app.is_running()`` loop and just calls ``read_command()``
each iteration, exactly like ``SOArmLeader.read_joint_state()``.

Key layout (WASD + QE for vertical, left/right arrow for yaw -- a standard
FPS-game layout, not a real UAV transmitter's stick layout):
  W / S           forward / backward  -> vx (+forward)
  A / D           strafe left / right -> vy (+left)
  E / Q           ascend / descend    -> vz (+up)      ("takeoff"/"land")
  Right / Left    yaw right / left    -> yaw_rate (about +Z)

The body frame here is FLU: **+x is forward**, i.e. the direction the nose
points, which on every preset carrying an arm is the direction that arm
reaches (measured: iris650_soarm100's eef sits at +0.301 m x / -0.311 m z
from base_link, amov950_airbot_bottom's at +0.297 / -0.344). +y is left, +z
is up. The caller rotates (vx, vy) by the current target yaw, so at yaw 0 a
held W moves the drone along world +x.

This used to be written the other way round -- W/S on +y and A/D on +x --
which put "forward" 90 deg off the nose: pressing W flew the robot sideways
and A/D was what moved it the way the arm was pointing.
Every axis is a simple held/not-held digital read (carb reports a keyboard
key as 0.0/1.0, not an analog stick) -- no response curve or deadzone
needed, each axis is just ``positive_key - negative_key`` in {-1, 0, 1}.
"""

import carb
import carb.input
import omni.appwindow

from .lerobot_motors.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

_K = carb.input.KeyboardInput

# (positive_key, negative_key) per axis.
_AXIS_KEYS = {
    "vx": (_K.W, _K.S),      # +x forward (nose/arm direction)
    "vy": (_K.A, _K.D),      # +y left
    "vz": (_K.E, _K.Q),      # +z up
    "yaw_rate": (_K.RIGHT, _K.LEFT),
}


class KeyboardUAVControl:
    """A connected keyboard, read as a normalized UAV flight-stick command."""

    def __init__(self):
        self._input = None
        self._keyboard = None
        self.connect()

    @property
    def is_connected(self) -> bool:
        return self._keyboard is not None

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("KeyboardUAVControl is already connected.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        print("KeyboardUAVControl connected: WASD move, E/Q up/down, Left/Right yaw.")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("KeyboardUAVControl is not connected.")
        self._input = None
        self._keyboard = None
        print("KeyboardUAVControl disconnected.")

    def _axis(self, positive_key, negative_key) -> float:
        pos = self._input.get_keyboard_value(self._keyboard, positive_key)
        neg = self._input.get_keyboard_value(self._keyboard, negative_key)
        return float(pos - neg)

    def read_command(self) -> dict:
        """Returns ``{"vx", "vy", "vz", "yaw_rate"}``, each in {-1, 0, 1}
        (body/target-frame intent, FLU: vx=forward, vy=left, vz=up, yaw_rate=
        +cw about +Z) -- feed into whichever flight-control mode the caller
        implements (see ``examples/aerial_manipulator_control_demo.py``'s
        ``--mode kinematic``/``dynamic``)."""
        if not self.is_connected:
            raise DeviceNotConnectedError("KeyboardUAVControl is not connected.")
        return {axis: self._axis(*keys) for axis, keys in _AXIS_KEYS.items()}
