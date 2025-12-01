"""
6-DoF gripper with its open/close variant
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        # Rigid gripper now has fixed fingers (no joints) like soft variant
        return np.array([])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class Robotiq85Gripper(Robotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        No-op for fixed fingers (no actuation).
        
        Note: Gripper fingers are now fixed (no actuators), so this returns empty array.

        Args:
            action (np.array): gripper-specific action (should be empty since dof=0)

        Returns:
            np.array: Empty array (no actuator commands)
        """
        # Fingers are fixed - no actuator commands needed
        return np.array([])

    @property
    def speed(self):
        return 0.20

    @property
    def dof(self):
        return 0  # Fixed fingers - no gripper actuation
