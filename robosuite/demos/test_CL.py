# the second script written to test simulation, open loop control of joints

import robosuite as suite
from robosuite.demos.demo_control import refactor_composite_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import GymWrapper
import time

if __name__ == "__main__":
    action_dim = 6
    control_freq = 20

    # Load the desired controller
    arm_controller_config = suite.load_part_controller_config(default_controller="OSC_POSITION")
    controller_configs = refactor_composite_controller_config(
        arm_controller_config, 'ur5e', ["right"]
    )

    use_depth = False
    peg_shape = None
    randomize_camera = False

    # initialize the task
    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            controller_configs=controller_configs,
            gripper_types='Robotiq85Gripper',  # 'Robotiq85GripperSoft', 'Robotiq85Gripper'
            has_renderer=True,
            initialization_noise=None,
            hard_reset=peg_shape == 'random',
            # ignore_done=True,
            has_offscreen_renderer=use_depth,
            camera_names="cam_view",
            use_camera_obs=use_depth,
            depth_mode='abs_m',
            camera_depths=use_depth,
            camera_widths=640,
            camera_heights=480,
            use_proprio_names=[
                # 'dummy_non_priv',
                # 'dummy_priv',
                'wrist_pos_rel',
                'wrist_force',
                'wrist_torque',
                'peg_pos_rel',
                'peg_rot6d',
                'peg_alignment',
            ],
            horizon=1000,
            control_freq=control_freq,
            deterministic_reset=False,
            hard_reset_every_n_episodes=1,
            success_reward=100,
            initial_pose=np.array([0.038, 0.665, 0.40, 1.0, 0.0, 0.0, 0.0]),
            peg_and_hole_color=dict(
                peg_default=[1.0, 1.0, 1.0],
                hole_default=[0.5, 0.5, 0.5],
                interp=0.2,
            ),
            shape=peg_shape,
            peg_pos_var=0.0,
            hole_pos_var=10,
            peg_angle_var=0,
            peg_friction_range=[1.0, 1.0],
            reward_type='baseline',
            camera_view_direction='real_calib',
            force_termination_threshold=30,
            renderer="mujoco",  # if peg_shape == 'random' else 'mjviewer',
            renderer_config={'cam_config': {"lookat": [0.3, 0.5, 0.38],
                                            "distance": 1.0, "azimuth": 180, "elevation": -25, }}
        ),
        flatten_obs=False,
    )
    cl_coef = 0.
    env.set_curriculum(cl_coef)
    obs, _ = env.reset()
    env.render()
    initial_pos = env.robots[0]._hand_pos['right']
    # env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Define neutral value
    assert len(env.robots) == 1
    # gripper_dim = env.robots[0].gripper.dof
    # neutral = np.zeros(action_dim + gripper_dim)
    neutral = np.array([1, 1, 1]) * 0.0
    # neutral = np.array([0, 0, -1])
    tic = time.time()
    diffs = []
    for i in range(10000):
        sec = i * 1.0 / control_freq

        action = np.zeros(env.action_space.shape[0])

        obs, rew, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

        if (i + 1) % 10 == 0:
            # final_pos = env.robots[0]._hand_pos['right']
            # diffs.append(final_pos - initial_pos)
            # # print(final_pos - initial_pos)
            # print(f"{env.peg_pos_error=}")
            env.set_curriculum(cl_coef)
            obs, info = env.reset()
            # initial_pos = env.robots[0]._hand_pos['right']

        if (i + 1) % 10 == 0:
            cl_coef += 0.02
            cl_coef = min(1.0, cl_coef)

            # print(f"Curriculum coefficient: {cl_coef:0.02f}")

    diff = np.mean(diffs, axis=0)
    print(f"{diff=}")

    env.close()
