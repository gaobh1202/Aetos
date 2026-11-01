"""Direct USB-serial reader for a lerobot SO-ARM leader arm (6x Feetech STS3215).

Adapted from leisaac's own ``SO101Leader``
(https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/devices/lerobot/so101_leader.py),
minus its ``Device``/``DeviceBase`` base class -- that base wires the leader
into IsaacLab's keyboard-event loop (``carb``/``omni.appwindow`` callbacks for
B/R/N keys), which this project doesn't use. ``examples/amteleop_soarm_leader.py``
drives its own ``while simulation_app.is_running()`` loop, so this class only
needs connect/calibrate/read -- the serial protocol itself
(``FeetechMotorsBus``) is vendored unchanged in ``aetos_multiam.devices.lerobot_motors``.
"""

import json
import os
import os.path as osp

from .lerobot_motors import FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode, OperatingMode
from .lerobot_motors.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# Motor ids/models/norm-modes are fixed by the leader's own hardware wiring
# (6x STS3215 on a shared bus) -- not something a USD or sim config chooses.
# Same layout as the follower's own joint set (so_arm100.py's
# arm_joint_names + the "gripper" joint), which is what makes a direct
# per-joint mirror (no IK) correct here.
_MOTOR_SPECS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

# The leader's own normalized reading range after calibration -- NOT the
# follower USD's joint range in degrees (see
# aetos_multiam.devices.action_convert.SO101_USD_JOINT_LIMITS_DEG for that).
# Matches leisaac's SO101_FOLLOWER_MOTOR_LIMITS exactly; this is a hardware/
# calibration-protocol constant (LeRobot's own RANGE_M100_100/RANGE_0_100
# norm modes), independent of which follower asset is on the other end.
MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


class SOArmLeader:
    """A connected SO-ARM leader arm, read as raw per-joint motor positions."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        recalibrate: bool = False,
        calibration_file_name: str = "soarm_leader.json",
    ):
        self.port = port
        self.calibration_path = osp.join(osp.dirname(__file__), ".cache", calibration_file_name)
        if not osp.exists(self.calibration_path) or recalibrate:
            self._run_calibration()
        calibration = self._load_calibration()

        self._bus = FeetechMotorsBus(port=self.port, motors=dict(_MOTOR_SPECS), calibration=calibration)
        self.motor_limits = MOTOR_LIMITS
        self.connect()

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("SOArmLeader is already connected.")
        self._bus.connect()
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        print(f"SOArmLeader connected on {self.port}.")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("SOArmLeader is not connected.")
        self._bus.disconnect()
        print("SOArmLeader disconnected.")

    def read_joint_state(self) -> dict:
        """Returns ``{joint_name: position}`` in the leader's own normalized
        units (see ``MOTOR_LIMITS``) -- feed into
        ``action_convert.convert_leader_state_to_radians`` before applying to
        the sim."""
        return self._bus.sync_read("Present_Position")

    def _run_calibration(self):
        bus = FeetechMotorsBus(port=self.port, motors=dict(_MOTOR_SPECS))
        bus.connect()
        print("\nRunning calibration of SOArmLeader")
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move SOArmLeader to the middle of its range of motion and press ENTER...")
        homing_offset = bus.set_half_turn_homings()
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = bus.record_ranges_of_motion()

        calibration = {}
        for motor, m in bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_path}")
        bus.disconnect()

    def _load_calibration(self) -> dict:
        with open(self.calibration_path) as f:
            data = json.load(f)
        return {
            name: MotorCalibration(
                id=int(v["id"]),
                drive_mode=int(v["drive_mode"]),
                homing_offset=int(v["homing_offset"]),
                range_min=int(v["range_min"]),
                range_max=int(v["range_max"]),
            )
            for name, v in data.items()
        }

    def _save_calibration(self, calibration: dict):
        payload = {
            name: {
                "id": cal.id,
                "drive_mode": cal.drive_mode,
                "homing_offset": cal.homing_offset,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
            }
            for name, cal in calibration.items()
        }
        os.makedirs(osp.dirname(self.calibration_path), exist_ok=True)
        with open(self.calibration_path, "w") as f:
            json.dump(payload, f, indent=2)
