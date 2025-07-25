"""
6-DoF gripper with its open/close variant
Mostly copied and pasted from robotiq_hand_e_gripper, but with different xml file and accounting for additional DOF from springs
(I really debated whether to make this a child class of RobotiqHandEGripper, but the xml import happens in the parent base class
so it seemed innappropriate to overwrite this in the child)
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class RobotiqHandEGripperSoftBase(GripperModel):
    """
    6-DoF Robotiq Hand-E gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_hand_e_soft.xml"), idn=idn)

    def format_action(self, action):
        return action

    # TODO: I init_qpos and _important_geoms arbitrarily

    @property
    def init_qpos(self):
        return_value = np.zeros(len(self.joints))
        # hard coded rest position of the vertical DOF spring
        # TODO: don't hard code this--it depends on the gripper mass and spring parameters
        return_value[0] = 0.008765
        return return_value

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
        }

    @property
    def _important_sensors(self):
        """
        Sensor names for each gripper (usually "force_ee" and "torque_ee")

        Returns:
            dict:

                :`'force_ee'`: Name of force eef sensor for this gripper
                :`'torque_ee'`: Name of torque eef sensor for this gripper
        """
        return {sensor: sensor for sensor in ["force_ee", "torque_ee", "force_peg", "torque_peg"]}


class RobotiqHandEGripperSoft(RobotiqHandEGripperSoftBase):
    """
    Copied and pasted from PandaGripper
    format_action from NullGripper
    """

    def format_action(self, action):
        return action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 0
