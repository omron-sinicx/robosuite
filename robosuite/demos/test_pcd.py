from argparse import ArgumentParser
import logging
import time
from pathlib import Path
from copy import deepcopy

import torch
import open3d as o3d
import mujoco
import robosuite as suite
# from robosuite.utils.transform_utils import *
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, transform_from_pixels_to_world
from robosuite.utils.pointcloud_utils import get_point_cloud_in_camera
from robosuite.wrappers import GymWrapper
from robosuite.demos.demo_control import refactor_composite_controller_config
from robosuite.utils.input_utils import *
from robosuite.utils.pointcloud_utils import furthest_point_sampling

import cv2
import numpy as np
np.set_printoptions(suppress=True)


def colorize_depth(depth, d_min=50, d_max=2000, d_scale=1000):
    depth = depth.copy() * d_scale
    depth = np.where(np.all((d_min < depth, depth < d_max), axis=0), depth, 0)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_8bit = cv2.convertScaleAbs(depth)
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return depth_colormap


def main(args):
    use_depth = True
    gripper_types = args.gripper
    peg_shape = args.shape

    # Load the desired controller
    arm_controller_config = suite.load_part_controller_config(default_controller="OSC_POSE")
    controller_configs = refactor_composite_controller_config(arm_controller_config, 'ur5e', ["right"])
    # print(controller_configs)

    hard_reset = (peg_shape == 'random') \
        or (peg_shape == 'random_char') \
        or (args.peg_size[0] != args.peg_size[1])
    # initialize the task
    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=None,
            has_renderer=True,
            hard_reset=hard_reset,
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
            control_freq=20,
            deterministic_reset=False,
            success_reward=100,
            initial_pose=np.array([0.0, 0.5, 0.28, -np.pi / 4, -np.pi / 4, 0.0, 0.0]),
            peg_shape=peg_shape,
            hole_pos_var=args.hole_pos_var,
            peg_angle_var=args.peg_angle_var,
            peg_friction_range=[args.peg_friction, args.peg_friction],
            peg_size_range=args.peg_size,
            peg_mass_range=[args.peg_mass, args.peg_mass],
            peg_distance_weights=np.ones(3),
            reward_type='baseline',
            camera_view_direction='real_calib',
            force_termination_threshold=None,
            renderer="mujoco" if hard_reset else 'mjviewer',
            renderer_config={'cam_config': {"lookat": [0.3, 0.5, 0.38],
                                            "distance": 1.0, "azimuth": 180, "elevation": -25, }}
        ),
        flatten_obs=False,
    )
    env.set_curriculum(args.curriculum)
    env.reset()
    env.viewer.set_camera(camera_id=0)
    # env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(env, pos_sensitivity=0.01, rot_sensitivity=0.0)
    env.viewer.add_keypress_callback(device.on_press)

    device.start_control()

    assert len(env.robots) == 1

    active_robot = env.robots[0]
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    prev_jpos = np.nan

    wrenchs = []
    for i in range(10000):
        # Get the newest action
        input_ac_dict = device.input2action()

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        obs, rew, terminated, truncated, info = env.step(env_action)
        env.render()

        img = obs['cam_view_image']
        depth = obs['cam_view_depth']
        if args.visualize:
            depth_color = colorize_depth(depth)
            cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.moveWindow('image', 0, 0)
            image_frame = cv2.getWindowImageRect('image')
            cv2.imshow('depth', depth_color)
            cv2.moveWindow('depth', 0, image_frame[3])
            cv2.waitKey(1)
            # print(depth.min(), depth.max(), depth.dtype)
            # saved_depth = (depth * 1000).astype('uint16')
            # print(saved_depth.min(), saved_depth.max(), saved_depth.dtype)
            # cv2.imwrite('depth.png', saved_depth)
            # break

        if not np.all(np.isclose(active_robot._joint_positions, prev_jpos, rtol=1e-5)):
            camera_name = 'cam_view'
            # Get camera intrinsic matrix
            camera_matrix = get_camera_intrinsic_matrix(
                env.sim,
                camera_name,
                env.camera_widths[0],
                env.camera_heights[0]
            )
            # print(camera_matrix)

            # Get camera pose (extrinsic matrix)
            # camera_pose = env.sim.data.get_camera_xmat(camera_name)
            # camera_pos = env.sim.data.get_camera_xpos(camera_name)
            # camera_pose = np.concatenate([camera_pose.reshape(3, 3), camera_pos.reshape(3, 1)], axis=1)
            # camera_pose = np.concatenate([camera_pose, np.array([[0, 0, 0, 1]])])
            # print(camera_pose)
            # if to_world:
            #     points_homogeneous = np.concatenate([pcl, np.ones((pcl.shape[0], 1))], axis=1)
            #     points_homogeneous = (camera_pose @ points_homogeneous.T).T
            #     pcl = points_homogeneous[:, :3]

            # print('depth range', depth.min(), depth.max())

            N_PTS = 1024
            print(depth[None, ..., 0].shape, camera_matrix[None, ...].shape)
            pcl = get_point_cloud_in_camera(depth[None, ..., 0], camera_matrix[None, ...])
            pcl = torch.from_numpy(pcl).float()
            # indices = torch.topk(pcl[:, 2, :], round(pcl.shape[-1] * 0.7), dim=-1, largest=False).indices
            # pcl = pcl.gather(2, indices.unsqueeze(1).expand(-1, 3, -1))
            # pcl = pcl.numpy()

            pcl_rnd = pcl[:, :, np.random.choice(pcl.shape[-1], N_PTS, replace=False)]
            # pcl_fps = furthest_point_sampling(torch.from_numpy(pcl.transpose(0, 2, 1)).float().contiguous().cuda(),
            #                                   N_PTS).cpu().numpy().transpose(0, 2, 1)

            # assert pcl_rnd.shape[-1] == pcl_fps.shape[-1] == N_PTS

            # Visualize using Open3D (optional)
            pcl_rnd = pcl_rnd.numpy()
            pcd_rnd = o3d.geometry.PointCloud()
            pcd_rnd.points = o3d.utility.Vector3dVector(pcl_rnd[0].T)
            pcd_rnd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (pcl_rnd.shape[-1], 1)))
            # pcd_fps = o3d.geometry.PointCloud()
            # pcd_fps.points = o3d.utility.Vector3dVector(pcl_fps[0].T)
            # pcd_fps.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (pcl_fps.shape[-1], 1)))
            axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_rnd, axis_frame])

        prev_jpos = active_robot._joint_positions

        if terminated or truncated:
            print('is_success', info['is_success'])
            print('is_truncated', info['is_truncated'])
            print("reset", f"{env.total_rewards=}")
            # input()
            obs, info = env.reset()

        time.sleep(0.03)

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    peg_dict = {
        '85': 'Robotiq85GripperSoft',
        'hande': 'RobotiqHandEGripperSoft',
    }

    parser = ArgumentParser()
    parser.add_argument('-g', '--gripper', default=list(peg_dict.keys())[0], choices=peg_dict.keys())
    parser.add_argument('-s', '--shape', default='triangle')
    parser.add_argument('-hpv', '--hole_pos_var', type=float, default=0.0)
    parser.add_argument('-pav', '--peg_angle_var', type=float, default=0.0)
    parser.add_argument('-pf', '--peg_friction', type=float, default=1.0)
    parser.add_argument('-ps', '--peg_size', nargs='+', type=float, default=[1.0, 1.0])
    parser.add_argument('-pm', '--peg_mass', type=float, default=0.03)
    parser.add_argument('-rc', '--randomize_camera', action='store_true', help='randomize camera')
    parser.add_argument('-viz', '--visualize', action='store_true', help='visualize camera obs')
    parser.add_argument('-cl', '--curriculum', type=float, default=0.0, help='curriculum coefficient')
    parser.add_argument('-tb', '--tensorboard', action='store_true', help='log to tensorboard')
    parser.add_argument('-log', '--log_level', type=str, default='debug')
    args = parser.parse_args()
    args.gripper = peg_dict[args.gripper]

    logging.basicConfig(level=args.log_level.upper())
    main(args)
