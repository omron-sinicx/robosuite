"""
End-effector for UR5e grinding task.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class UR5eGrinder(GripperModel):
    """
    End-effector with no actuation for UR5e grinding task.

    Args:
        idn (int or str): Number or some other unique identification string for this eef instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/soft_pestle_jig.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return_value = np.zeros(len(self.joints))
        return return_value

    @property
    def _important_geoms(self):
        return {
            "eef": ["pestle_collision"],
        }
