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

    gripper_types = "Robotiq85GripperSoft"
    use_depth = False
    peg_shape = 'round'

    # initialize the task
    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=None,
            has_renderer=False,
            hard_reset=peg_shape == 'random',
            ignore_done=True,
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
            success_reward=100,
            initial_pose=np.array([0.0, 0.65, 0.28, -np.pi / 4, -np.pi / 4, 0.0, 0.0]),
            peg_shape=peg_shape,
            hole_pos_var=0,
            peg_angle_var=0,
            peg_friction_range=[1, 1],
            reward_type='baseline',
            camera_view_direction='real_calib',
            force_termination_threshold=30,
            renderer="mujoco" if peg_shape == 'random' else 'mjviewer',
            renderer_config={'cam_config': {"lookat": [0.3, 0.5, 0.38],
                                            "distance": 1.0, "azimuth": 180, "elevation": -25, }}
        ),
        flatten_obs=False,
    )
    obs, _ = env.reset()
    initial_pos = env.robots[0]._hand_pos['right']
    # env.viewer.set_camera(camera_id=3)
    # env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Define neutral value
    assert len(env.robots) == 1
    # gripper_dim = env.robots[0].gripper.dof
    # neutral = np.zeros(action_dim + gripper_dim)
    neutral = np.array([1, 1, 1])

    tic = time.time()
    diffs = []
    for i in range(1000):
        sec = i * 1.0 / control_freq

        action = neutral.copy()
        # action_idx = int(sec) % 6  # 6 DOF action
        # action_sign = 2.0 * (int(sec) % 12 < 5) - 1  # alternate directions
        # action_value = (
        #     0.49 * (int(sec) % 6 > 2) + 0.01
        # )  # larger values for radian angles
        # action[action_idx] = action_sign * action_value

        # action[3] = 0.2 * np.sin(sec)
        # action[-1] = -1.0 + 2.0 * (int(sec) % 2 == 0)

        # action[2] += 0.05 * np.sin(sec)

        obs, rew, terminated, truncated, info = env.step(action)
        # env.render()
        # import ipdb; ipdb.set_trace()

        # from scipy.spatial.transform import Rotation
        # ori = Rotation.from_quat(obs[3:])
        # rot_180_x = Rotation.from_quat([0.0, 1.0, 0.0, 0.0])
        # print((rot_180_x * ori).as_euler("xyz"))

        # if terminated or truncated:
        if (i + 1) % 10 == 0:
            final_pos = env.robots[0]._hand_pos['right']
            diffs.append(final_pos - initial_pos)
            # print(final_pos - initial_pos)
            obs, info = env.reset()
            initial_pos = env.robots[0]._hand_pos['right']

        # # sleep to make animation realtime
        # toc = time.time() - tic
        # time.sleep(max(1.0/control_freq - toc, 0))
        # tic = time.time()
    diff = np.mean(diffs, axis=0)
    print(f"{diff=}")

    env.close()
