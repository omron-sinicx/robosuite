"""Driver class for GELLO device using Dynamixel motors.

This module provides a teleoperation interface using GELLO (a physical 
robot arm with Dynamixel motors) to control robot manipulators in simulation.
"""

import threading
import time
from typing import Dict, Optional
import numpy as np
from pynput.keyboard import Key, Listener

try:
    import sys
    import os
    # Add the Dynamixel SDK to the Python path
    dynamixel_sdk_path = '/root/osx-ur/underlay_ws/src/third_party/DynamixelSDK/python/src'
    if dynamixel_sdk_path not in sys.path:
        sys.path.append(dynamixel_sdk_path)

    from dynamixel_sdk import (
        PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite,
        COMM_SUCCESS, COMM_TX_FAIL
    )
except ImportError as exc:
    raise ImportError(
        "Unable to load dynamixel_sdk module, required to interface with GELLO device. "
        "Make sure the DynamixelSDK is properly installed and accessible."
    ) from exc

from robosuite.devices.device import Device
from robosuite.utils.transform_utils import rotation_matrix


class GELLO(Device):
    """
    A driver class for GELLO device using Dynamixel motors.

    The GELLO device reads joint positions from Dynamixel motors and converts them
    to robot control commands. It assumes the motors are ordered correctly to match
    the simulated robot's joint order.

    Args:
        env: The robosuite environment
        device_name (str): USB device name (e.g., "/dev/ttyUSB0")
        motor_ids (list): List of Dynamixel motor IDs in joint order
        baudrate (int): Communication baudrate
        protocol_version (float): Dynamixel protocol version
        pos_sensitivity (float): Position sensitivity scaling factor
        rot_sensitivity (float): Rotation sensitivity scaling factor
        motor_zero_positions (list): Motor positions corresponding to joint angle 0 for each motor
    """

    def __init__(
        self,
        env,
        device_name="/dev/ttyUSB0",
        joint_motor_ids=[1, 2, 3, 4, 5, 6],  # Default: 6 arm joints + 1 gripper
        gripper_motor_id=None,
        baudrate=4000000,
        protocol_version=2.0,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        joint_limits=None,
        motor_zero_positions=None,
        motor_direction=None,
    ):
        super().__init__(env)

        # GELLO/Dynamixel configuration
        self.device_name = device_name.encode('utf-8') if isinstance(device_name, str) else device_name
        self.joint_motor_ids = joint_motor_ids
        self.arm_motor_ids = joint_motor_ids[:-1]  # All except last (gripper)
        self.gripper_motor_id = gripper_motor_id  # Last motor is gripper
        self.joint_num = len(joint_motor_ids)
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.motor_direction = np.array(motor_direction) if motor_direction is not None else np.ones(len(joint_motor_ids))

        self._reset_internal_state()

        # Control sensitivity
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # Dynamixel control table addresses (for X-series motors)
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PRESENT_POSITION = 132
        self.LEN_PRESENT_POSITION = 4

        # Joint limits for scaling (if not provided, use reasonable defaults)
        if joint_limits is None:
            # Default limits for typical robot arm (in radians)
            self.joint_limits = [
                [-np.pi, np.pi],      # Joint 1
                [-np.pi, np.pi],  # Joint 2
                [-np.pi, np.pi],      # Joint 3
                [-np.pi, np.pi],      # Joint 4
                [-np.pi, np.pi],      # Joint 5
                [-2*np.pi, 2*np.pi],      # Joint 6
                [0.0, 1.0],           # Gripper (0=open, 1=closed)
            ]
        else:
            self.joint_limits = joint_limits

        # Motor zero positions (motor positions that correspond to joint angle 0)
        if motor_zero_positions is None:
            # Default zero positions - you should calibrate these for your setup
            self.motor_zero_positions = [
                2048,  # Motor 1 zero position (middle of range)
                2048,  # Motor 2 zero position
                2048,  # Motor 3 zero position
                2048,  # Motor 4 zero position
                2048,  # Motor 5 zero position
                2048,  # Motor 6 zero position
                0,     # Gripper zero position (fully open)
            ]
            print("WARNING: Using default motor zero positions. Please calibrate for your setup!")
            print("See calibration instructions in the documentation.")
        else:
            self.motor_zero_positions = motor_zero_positions

        # Initialize hardware connection
        self._init_dynamixel_connection()

        # State variables
        self.joint_positions = np.zeros(self.joint_num)
        self.previous_joint_positions = np.zeros(self.joint_num)
        self.control_gripper = 0.0
        self._reset_state = 0
        self._enabled = False
        self._running = True  # Flag to control thread execution

        # Button state for keyboard controls (X, Y, A, B equivalent)
        self._btn_state = [0, 0, 0, 0]  # state of equivalent X, Y, A, B buttons

        # Setup keyboard listener
        self._setup_keyboard_listener()

        # Launch listener thread
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # Start keyboard listener
        self.keyboard_listener.start()

    def _init_dynamixel_connection(self):
        """Initialize connection to Dynamixel motors."""
        # Initialize PortHandler and PacketHandler
        self.port_handler = PortHandler(self.device_name.decode('utf-8'))
        self.packet_handler = PacketHandler(self.protocol_version)

        # Open port
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.device_name}")
        print(f"Successfully opened port {self.device_name}")

        # Set baudrate
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate to {self.baudrate}")
        print(f"Successfully set baudrate to {self.baudrate}")

        # Initialize GroupSyncRead for efficient multi-motor reading
        self.group_sync_read = GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            self.ADDR_PRESENT_POSITION,
            self.LEN_PRESENT_POSITION
        )

        # Add all motors to sync read group
        for motor_id in self.joint_motor_ids:
            if not self.group_sync_read.addParam(motor_id):
                raise RuntimeError(f"Failed to add motor {motor_id} to sync read group")

        print(f"Successfully initialized GELLO with motors: {self.joint_motor_ids}")
        self._display_keyboard_controls()

    def _dynamixel_to_radians(self, motor_value, joint_idx):
        """Convert Dynamixel position value to radians based on joint limits and zero position."""
        # Get the zero position for this motor
        motor_zero = self.motor_zero_positions[joint_idx]

        # Calculate offset from zero position
        motor_offset = motor_value - motor_zero

        # Dynamixel X-series typically uses 0-4095 for full rotation (about 2π radians)
        motor_range = 4095
        radians_per_count = (2 * np.pi) / motor_range

        # Convert to radians
        joint_angle_raw = motor_offset * radians_per_count

        # Apply joint limits
        joint_min, joint_max = self.joint_limits[joint_idx]
        joint_angle = np.clip(joint_angle_raw, joint_min, joint_max)

        return joint_angle

    def _read_joint_positions(self):
        """Read current joint positions from all Dynamixel motors."""
        try:
            # Check if we should still be running
            if not self._running:
                return False

            # Sync read all motors
            comm_result = self.group_sync_read.txRxPacket()
            if comm_result != COMM_SUCCESS:
                if self._running:  # Only print error if we're still supposed to be running
                    print(f"Sync read failed: {self.packet_handler.getTxRxResult(comm_result)}")
                return False

            # Extract positions for each motor
            positions = []
            for i, motor_id in enumerate(self.joint_motor_ids):
                if not self._running:  # Check again in the loop
                    return False

                if not self.group_sync_read.isAvailable(motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                    if self._running:
                        print(f"Data not available for motor {motor_id}")
                    return False

                # Get raw position value
                raw_position = self.group_sync_read.getData(motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

                # Check for invalid readings (handle potential negative values from unsigned conversion)
                if raw_position > 2**31:  # Likely a negative value interpreted as unsigned
                    raw_position = raw_position - 2**32

                # Clamp to reasonable range for Dynamixel (0-4095)
                raw_position = max(0, min(4095, raw_position))

                # Convert to radians (or normalized value for gripper)
                joint_position = self._dynamixel_to_radians(raw_position, i)

                positions.append(joint_position)

            if self.gripper_motor_id is not None:
                if not self._running:  # Check again in the loop
                    return False

                if not self.group_sync_read.isAvailable(self.gripper_motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                    if self._running:
                        print(f"Data not available for motor {self.gripper_motor_id}")
                    return False

                # Get raw position value
                raw_position = self.group_sync_read.getData(self.gripper_motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

                # Check for invalid readings (handle potential negative values from unsigned conversion)
                if raw_position > 2**31:  # Likely a negative value interpreted as unsigned
                    raw_position = raw_position - 2**32

                # Clamp to reasonable range for Dynamixel (0-4095)
                raw_position = max(0, min(4095, raw_position))
                gripper_zero = self.motor_zero_positions[-1]
                gripper_offset = raw_position - gripper_zero
                # Normalize to 0-1 range (assuming 0-4095 range from zero position)
                joint_position = max(0.0, min(1.0, gripper_offset / 4095.0))

                positions.append(joint_position)

            self.joint_positions = np.array(positions) * self.motor_direction
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Handle any exceptions that might occur during communication
            if self._running:  # Only print error if we're still supposed to be running
                print(f"Error reading joint positions: {e}")
            return False

    def _reset_internal_state(self):
        """Reset internal state of controller."""
        super()._reset_internal_state()

        self.joint_positions = np.zeros(self.joint_num)
        self.previous_joint_positions = np.zeros(self.joint_num)
        self.control_gripper = 0.0
        self._btn_state = [0, 0, 0, 0]  # Reset button states

    def start_control(self):
        """Start control by enabling the device."""
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Get the current state of the GELLO device.

        Returns:
            dict: A dictionary containing dpos, rotation, raw_drotation, grasp, and reset
        """
        if not self._enabled:
            return {
                "grasp": 0.0,
                "reset": self._reset_state,
                "joint_positions": self.joint_positions.copy(),
                "buttons_state": self._btn_state.copy(),
            }

        return {
            "grasp": self.control_gripper,
            "reset": self._reset_state,
            "joint_positions": self.joint_positions.copy(),
            "buttons_state": self._btn_state.copy(),
        }

    def run(self):
        """Main loop that continuously reads joint positions."""
        print("GELLO listener thread started")

        try:
            while self._running:
                if self._enabled and self._running:
                    # Store previous positions
                    self.previous_joint_positions = self.joint_positions.copy()

                    # Read new positions
                    if not self._read_joint_positions():
                        if self._running:  # Only sleep if we're still supposed to be running
                            time.sleep(0.01)  # Short sleep on read failure
                        continue

                if self._running:  # Only sleep if we're still supposed to be running
                    time.sleep(0.01)  # 100Hz update rate

        except Exception as e:
            if self._running:  # Only print error if we're still supposed to be running
                print(f"Error in GELLO thread: {e}")
        finally:
            print("GELLO listener thread stopped")

    def _setup_keyboard_listener(self):
        """Setup keyboard listener for button controls."""
        self.keyboard_listener = Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )

    def _display_keyboard_controls(self):
        """Display keyboard controls for GELLO device."""
        print("")
        print("GELLO Keyboard Controls:")
        print("  j key: Toggle stiffness (equivalent to gamepad X button)")
        print("  y key: Stop recording (equivalent to gamepad Y button)")
        print("  k key: Save episode (equivalent to gamepad A button)")
        print("  p key: Reserved (equivalent to gamepad B button)")
        print("  o key: Reset simulation")
        print("")

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                if char == 'j':
                    self._btn_state[0] = 1  # X button equivalent
                elif char == 'y':
                    self._btn_state[1] = 1  # Y button equivalent
                elif char == 'o':
                    self._btn_state[2] = 1  # A button equivalent
                elif char == 'm':
                    self._btn_state[3] = 1  # B button equivalent
                elif char == 'k':
                    self._reset_state = 1
                    self._enabled = False
        except AttributeError:
            pass

    def _on_key_release(self, key):
        """Handle key release events."""
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                if char == 'j':
                    self._btn_state[0] = 0  # X button equivalent
                elif char == 'y':
                    self._btn_state[1] = 0  # Y button equivalent
                elif char == 'o':
                    self._btn_state[2] = 0  # A button equivalent
                elif char == 'm':
                    self._btn_state[3] = 0  # B button equivalent
        except AttributeError:
            pass

    def close(self):
        """Clean up and close connections."""
        print("Closing GELLO connection...")

        # Stop the background thread first
        self._running = False
        self._enabled = False

        # Stop keyboard listener
        if hasattr(self, 'keyboard_listener'):
            try:
                self.keyboard_listener.stop()
            except Exception as e:
                print(f"Error stopping keyboard listener: {e}")

        # Wait for thread to finish (with timeout)
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("Warning: GELLO thread did not stop cleanly")

        # Clean up resources
        try:
            if hasattr(self, 'group_sync_read'):
                self.group_sync_read.clearParam()
        except Exception as e:
            print(f"Error clearing sync read params: {e}")

        try:
            if hasattr(self, 'port_handler') and self.port_handler:
                self.port_handler.closePort()
        except Exception as e:
            print(f"Error closing port: {e}")

        print("GELLO connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup in destructor

    @property
    def control(self):
        """
        Get current control values.

        Returns:
            np.array: 6-DOF control value
        """
        return np.array(self._control)

    def input2action(self, mirror_actions=False) -> Optional[Dict]:
        """
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

        If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.

        Returns:
            Optional[Dict]: Dictionary of actions to be fed into env.step()
                            if reset is triggered, returns None
        """
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        state = self.get_controller_state()
        # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
        #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
        #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
        grasp, reset, joint_positions = (
            state["grasp"],
            state["reset"],
            state["joint_positions"],
        )

        # If we're resetting, immediately return None
        if reset:
            return None

        # Get controller reference
        controller = robot.part_controllers[active_arm]
        gripper = robot.gripper[active_arm]
        gripper_dof = robot.gripper[active_arm].dof

        assert controller.name in ["OSC_POSE", "JOINT_POSITION"], "only supporting OSC_POSE and JOINT_POSITION for now"

        # map 0 to -1 (open) and map 1 to 1 (closed)
        grasp = 1 if grasp else -1

        ac_dict = {}
        ac_dict[f"{active_arm}_joint_abs"] = joint_positions
        ac_dict[f"{active_arm}_joint_delta"] = joint_positions - self.previous_joint_positions

        if hasattr(gripper, "grasp_qpos"):
            ac_dict[f"{active_arm}_gripper"] = getattr(gripper, "grasp_qpos")[grasp]
        else:
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)

        # clip actions between -1 and 1
        for (k, v) in ac_dict.items():
            if "abs" not in k and "gripper" not in k and v is not None:
                ac_dict[k] = np.clip(v, -1, 1)

        ac_dict["state"] = state
        return ac_dict
