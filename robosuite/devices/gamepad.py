"""Driver class for GamePad controller .


"""

import threading
import time
from collections import namedtuple

import numpy as np
from robosuite import macros

try:
    from inputs import devices
    from inputs import UnpluggedError
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module inputs, required to interface with GamePad. "
        "Install the additional requirements with "
        "`pip install -r requirements-extra.txt`"
    ) from exc

from robosuite.devices.device import Device
from robosuite.utils.transform_utils import rotation_matrix

BUTTONS_INFO = {
    'ABS_HAT0X':  {'max': 1,     'min': -1,     'desc': 'Left pad horizontal'},
    'ABS_HAT0Y':  {'max': 1,     'min': -1,     'desc': 'Left pad vertical'},
    'ABS_X':      {'max': 32767, 'min': -32768, 'desc': 'Left stick horizontal'},
    'ABS_Y':      {'max': 32767, 'min': -32768, 'desc': 'Left stick vertical'},
    'ABS_RX':     {'max': 32767, 'min': -32768, 'desc': 'Right stick horizontal'},
    'ABS_RY':     {'max': 32767, 'min': -32768, 'desc': 'Right stick vertical'},
    'ABS_Z':      {'max': 255,   'min': 0,      'desc': 'LT'},
    'ABS_RZ':     {'max': 255,   'min': 0,      'desc': 'RT'},
    'BTN_THUMBL': {'max': 1,     'min': 0,      'desc': 'Left stick button'},
    'BTN_THUMBR': {'max': 1,     'min': 0,      'desc': 'Right stick button'},
    'BTN_NORTH':  {'max': 1,     'min': 0,      'desc': 'X'},
    'BTN_WEST':   {'max': 1,     'min': 0,      'desc': 'Y'},
    'BTN_SOUTH':  {'max': 1,     'min': 0,      'desc': 'A'},
    'BTN_EAST':   {'max': 1,     'min': 0,      'desc': 'B'},
    'BTN_TL':     {'max': 1,     'min': 0,      'desc': 'LB'},
    'BTN_TR':     {'max': 1,     'min': 0,      'desc': 'RB'},
    'BTN_SELECT': {'max': 1,     'min': 0,      'desc': 'BACK'},
    'BTN_START':  {'max': 1,     'min': 0,      'desc': 'START'},
}

AxisSpec = namedtuple("AxisSpec", ["direction", "range", "scale"])

# GAMEPAD_SPEC = {
#     'ABS_HAT0X': '',
#     'ABS_HAT0Y': '',
#     'ABS_X': AxisSpec(direction=1, range=[-1, 1], scale=1),
#     'ABS_Y': AxisSpec(direction=0, range=[-1, 1], scale=1),
#     'ABS_RX': AxisSpec(direction=4, range=[-1, 1], scale=1),
#     'ABS_RY': AxisSpec(direction=3, range=[-1, 1], scale=1),
#     'ABS_Z': AxisSpec(direction=2, range=[-1, 0], scale=-1),
#     'ABS_RZ': AxisSpec(direction=2, range=[0, 1], scale=1),
#     'BTN_THUMBL': '',
#     'BTN_THUMBR': '',
#     'BTN_NORTH': '',
#     'BTN_WEST': '',
#     'BTN_SOUTH': 'gripper',
#     'BTN_EAST': '',
#     'BTN_TL': AxisSpec(direction=5, range=[-1, 0], scale=-1),
#     'BTN_TR': AxisSpec(direction=5, range=[0, 1], scale=1),
#     'BTN_SELECT': '',
#     'BTN_START': 'reset',
# }

# GAMEPAD_SPEC = {
#     'ABS_HAT0X': '',
#     'ABS_HAT0Y': '',
#     'ABS_X': AxisSpec(direction=1, range=[-1, 1], scale=1),
#     'ABS_Y': AxisSpec(direction=2, range=[-1, 1], scale=-1),
#     'ABS_RX': AxisSpec(direction=4, range=[-1, 1], scale=1),
#     'ABS_RY': AxisSpec(direction=5, range=[-1, 1], scale=1),
#     'ABS_Z': AxisSpec(direction=0, range=[-1, 0], scale=-1),
#     'ABS_RZ': AxisSpec(direction=0, range=[0, 1], scale=1),
#     'BTN_THUMBL': '',
#     'BTN_THUMBR': '',
#     'BTN_NORTH': '',
#     'BTN_WEST': '',
#     'BTN_SOUTH': 'gripper',
#     'BTN_EAST': '',
#     'BTN_TL': AxisSpec(direction=3, range=[-1, 0], scale=-1),
#     'BTN_TR': AxisSpec(direction=3, range=[0, 1], scale=1),
#     'BTN_SELECT': '',
#     'BTN_START': 'reset',
# }

# GAMEPAD_SPEC = {
#     'ABS_HAT0X': AxisSpec(direction=4, range=[-1, 1], scale=1),
#     'ABS_HAT0Y': AxisSpec(direction=3, range=[-1, 1], scale=1),
#     'ABS_X': AxisSpec(direction=1, range=[-1, 1], scale=1),
#     'ABS_Y': AxisSpec(direction=2, range=[-1, 1], scale=-1),
#     'ABS_RX': '',
#     'ABS_RY': AxisSpec(direction=0, range=[-1, 1], scale=1),
#     'ABS_Z': AxisSpec(direction=5, range=[-1, 0], scale=-1),
#     'ABS_RZ': AxisSpec(direction=5, range=[0, 1], scale=1),
#     'BTN_THUMBL': '',
#     'BTN_THUMBR': '',
#     'BTN_NORTH': 'X',
#     'BTN_WEST': 'Y',
#     'BTN_SOUTH': 'A',
#     'BTN_EAST': 'B',
#     'BTN_TL': 'gripper',
#     'BTN_TR': 'gripper',
#     'BTN_SELECT': '',
#     'BTN_START': 'reset',
# }

GAMEPAD_SPEC = {
    'ABS_HAT0X': AxisSpec(direction=4, range=[-1, 1], scale=1),
    'ABS_HAT0Y': AxisSpec(direction=3, range=[-1, 1], scale=1),
    'ABS_X': AxisSpec(direction=1, range=[-1, 1], scale=-1),
    'ABS_Y': AxisSpec(direction=0, range=[-1, 1], scale=-1),
    'ABS_RX': '',
    'ABS_RY': AxisSpec(direction=2, range=[-1, 1], scale=-1),
    'ABS_Z': AxisSpec(direction=5, range=[-1, 0], scale=-1),
    'ABS_RZ': AxisSpec(direction=5, range=[0, 1], scale=1),
    'BTN_THUMBL': '',
    'BTN_THUMBR': '',
    'BTN_NORTH': 'X',
    'BTN_WEST': 'Y',
    'BTN_SOUTH': 'A',
    'BTN_EAST': 'B',
    'BTN_TL': 'gripper_on',
    'BTN_TR': 'gripper_off',
    'BTN_SELECT': 'back',
    'BTN_START': 'reset',
}


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    # print(x, axis_scale, min_v, max_v)
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    # print(x)
    return x


class GamePad(Device):
    """
    A minimalistic driver class for SpaceMouse with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.

    Args:
        vendor_id (int): HID device vendor id
        product_id (int): HID device product id
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(
        self,
        env,
        device_name=macros.GAMEPAD_NAME,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        deadzone=0.05,
    ):
        super().__init__(env)

        self._reset_internal_state()

        if len(devices.gamepads) == 0:
            raise UnpluggedError("No gamepad found.")

        # device_index = -1
        # for device in devices.gamepads:
        #     if device.name == device_name:
        #         print(f"Connecting to device: {device.name}")
        #         device_index = device.get_number()

        # if device_index == -1:
        #     raise ValueError(f"Gamepad '{device_name}' not found. Input a valid name.")

        # self.gamepad = devices.gamepads[device_index]

        self.gamepad = None
        for device in devices.gamepads:
            if device.name == device_name:
                print(f"Connecting to device: {device.name}")
                self.gamepad = device
                break

        if self.gamepad is None:
            raise ValueError(f"Gamepad '{device_name}' not found. Input a valid name.")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self.deadzone = deadzone

        self.control_gripper = 0.0

        # self._display_controls()

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self._btn_state = [0, 0, 0, 0]  # state of X, Y, A, B buttons
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def _postprocess_device_outputs(self, dpos, drotation):
        return dpos, drotation

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
            buttons_state=self._btn_state,
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        while True:
            events = self.gamepad.read()
            for event in events:
                if event.ev_type in ['Absolute', 'Key'] and event.code in BUTTONS_INFO:

                    # For each button. As long as the button/Joystick are being active, repeat the event, to get continuous motion.
                    btn = GAMEPAD_SPEC[event.code]
                    if isinstance(btn, AxisSpec):
                        scaled_input = scale_to_control(event.state*btn.scale, BUTTONS_INFO[event.code]['max'], min_v=btn.range[0], max_v=btn.range[1])
                        self._control[btn.direction] = scaled_input if abs(scaled_input) > self.deadzone else 0.0
                    elif 'gripper' in btn:  # Only trigger on button release
                        self.control_gripper = 1 if btn == 'gripper_on' else 0
                    elif btn == 'reset':
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()
                    elif btn == 'back' and event.state == 0:  # Only trigger on button release
                        self.active_robot = (self.active_robot + 1) % self.num_robots
                    elif btn == 'X':
                        self._btn_state[0] = int(event.state)
                    elif btn == 'Y':
                        self._btn_state[1] = int(event.state)
                    elif btn == 'A':
                        self._btn_state[2] = int(event.state)
                    elif btn == 'B':
                        self._btn_state[3] = int(event.state)

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)


if __name__ == "__main__":

    gamepad = GamePad(device_name="Logitech Gamepad F310")

    for i in range(100):
        print(gamepad.control, gamepad.control_gripper)
        time.sleep(0.02)
