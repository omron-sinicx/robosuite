"""
6-DoF gripper with its open/close variant
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Robotiq85GripperSoftBase(GripperModel):
    """
    6-DoF Robotiq 85 gripper with the soft wrist.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85_soft.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return_value = np.zeros(len(self.joints))
        # hard coded rest position of the vertical DOF spring
        # TODO: don't hard code this--it depends on the gripper mass and spring parameters
        return_value[0] = 0.00988
        return return_value
        # return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                # "left_outer_finger_collision",
                # "left_inner_finger_collision",
                # "left_fingertip_collision",
                # "left_fingerpad_collision",
            ],
            "right_finger": [
                # "right_outer_finger_collision",
                # "right_inner_finger_collision",
                # "right_fingertip_collision",
                # "right_fingerpad_collision",
            ],
            "left_fingerpad": [
                # "left_fingerpad_collision"
            ],
            "right_fingerpad": [
                # "right_fingerpad_collision"
            ],
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


class Robotiq85GripperSoft(Robotiq85GripperSoftBase):

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 0
