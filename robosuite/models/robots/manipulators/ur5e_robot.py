import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class UR5e(ManipulatorModel):
    """
    UR5e is a sleek and elegant new robot created by Universal Robots

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def init_qpos(self):
        # return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])
        return np.pi / 2.0 * np.array([1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
        # hard coded joint positions with peg roughly above the hole
        # return np.pi / 180.0 * np.array([92.69, -65.35, 80.49, -105.11, -89.48, -89.50])
        # return np.array(
        #     [1.62270375, -1.12578612, 1.44075275, -1.89047942, -1.58394959, -1.51093514]
        # )  # temp: peg partially inserted

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
