# the first script written to test simulation. GUI-based control of torques

import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
import mujoco
import time

import mujoco.viewer

if __name__ == "__main__":
    # use the env just as a convenient to create the a mujoco.MjModel object
    env = suite.make(
        env_name='Stack',
        robots='UR5e',
        # gripper_types=None,
        gripper_types="RobotiqHandEGripper",
        controller_configs=load_controller_config(default_controller="IK_POSE"),
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        horizon=1000,
        control_freq=10,
    )
    env.reset()

    mujoco_model = env.sim.model._model
    mujoco_data = mujoco.MjData(mujoco_model)

    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
        for i in range(1000000):
            mujoco.mj_step(mujoco_model, mujoco_data)
            viewer.sync()
