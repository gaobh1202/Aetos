"""Keyboard-polled per-joint manipulator input device.

The joint-space counterpart to ``keyboard_uav_control.KeyboardUAVControl``,
and the fallback arm teleop for every preset that has no dedicated hardware
yet. Same carb polling API and same read-per-frame contract (see that
module's docstring for why this project polls instead of registering
callbacks).

A keyboard has nowhere near enough keys for one pair per DOF (airbot alone
has eight), so this is a *selector*: number keys pick the active joint, one
key pair moves it. AM-Bench's own keyboard teleop has the same shape
constraint and spends it differently -- it drives a 6-DOF end-effector delta
and lets IK resolve the joints, which is smoother to fly but cannot command
a single joint. Per-joint is what an operator needs for checking a new
robot's joint limits, signs, and drive health.

Key layout (chosen not to collide with the flight stick's WASD/QE/arrows):
  1 .. 9          select the active joint (prints its name)
  ] / [  or  Up / Down    move the active joint  +/-
  G / H           close / open every gripper DOF at once
  Z               send every joint back to zero
Each axis is a digital held/not-held read, so the caller integrates:
``read_command()`` returns a *velocity* in joint units per second, scaled by
the caller's own rate.
"""

import carb
import carb.input
import omni.appwindow

from .lerobot_motors.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

_K = carb.input.KeyboardInput

_SELECT_KEYS = (_K.KEY_1, _K.KEY_2, _K.KEY_3, _K.KEY_4, _K.KEY_5,
                _K.KEY_6, _K.KEY_7, _K.KEY_8, _K.KEY_9)
# Two keys per action, because `[`/`]` are not where a US layout puts them on
# every keyboard (AltGr+8/9 on a German one, dead keys on a French one) and
# carb reports the physical key, not the character. Up/Down are layout-proof
# and sit next to the Left/Right the flight stick already uses.
_JOINT_MINUS = (_K.LEFT_BRACKET, _K.DOWN)
_JOINT_PLUS = (_K.RIGHT_BRACKET, _K.UP)
_GRIPPER_CLOSE, _GRIPPER_OPEN = (_K.G,), (_K.H,)
_ZERO_ALL = (_K.Z,)


class KeyboardArmControl:
    """A connected keyboard, read as a per-joint manipulator command."""

    def __init__(self, joint_names):
        self.joint_names = list(joint_names)
        self.active = 0
        self._input = None
        self._keyboard = None
        self._last_active = -1
        self._saw_a_key = False
        self.connect()

    @property
    def is_connected(self) -> bool:
        return self._keyboard is not None

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("KeyboardArmControl is already connected.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        print(
            "KeyboardArmControl connected: "
            f"1-{min(9, len(self.joint_names))} select joint, ] / [ (or Up / Down) move it, "
            "G/H gripper close/open, Z zero all.\n"
            "  joints: "
            + ", ".join(f"{i + 1}={n}" for i, n in enumerate(self.joint_names[:9]))
            + (" (joints past 9 are selectable only via the other devices)"
               if len(self.joint_names) > 9 else "")
        )

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("KeyboardArmControl is not connected.")
        self._input = None
        self._keyboard = None

    def _held(self, keys) -> float:
        if not isinstance(keys, tuple):
            keys = (keys,)
        return float(any(self._input.get_keyboard_value(self._keyboard, k) for k in keys))

    def read_command(self) -> dict:
        """Returns ``{"active", "joint_rate", "gripper_rate", "zero_all"}``.

        ``joint_rate``/``gripper_rate`` are in {-1, 0, 1} and mean "joint
        units per second" once the caller multiplies by its own dt and speed;
        ``active`` is the index into ``joint_names`` the rate applies to.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("KeyboardArmControl is not connected.")
        for i, key in enumerate(_SELECT_KEYS[:len(self.joint_names)]):
            if self._held(key):
                self.active = i
        if self.active != self._last_active:
            print(f"[keyboard_arm] active joint: {self.active + 1}="
                  f"{self.joint_names[self.active]}", flush=True)
            self._last_active = self.active
        command = {
            "active": self.active,
            "joint_rate": self._held(_JOINT_PLUS) - self._held(_JOINT_MINUS),
            "gripper_rate": self._held(_GRIPPER_OPEN) - self._held(_GRIPPER_CLOSE),
            "zero_all": self._held(_ZERO_ALL) > 0.0,
        }
        # Say so the first time a key actually registers. "The arm doesn't
        # move" has two very different causes -- the key never arrived, or the
        # write didn't land -- and they look identical from the viewport.
        if not self._saw_a_key and (command["joint_rate"] or command["gripper_rate"]
                                    or command["zero_all"]):
            print("[keyboard_arm] first key registered -- input is reaching the sim", flush=True)
            self._saw_a_key = True
        return command
