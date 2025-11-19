# the third script written to test simulation, control via keyboard
from argparse import ArgumentParser
import logging
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
np.set_printoptions(suppress=True)
import cv2
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import mujoco
from robosuite.utils.transform_utils import *
from robosuite.wrappers import GymWrapper
from robosuite.demos.demo_control import refactor_composite_controller_config
from robosuite.utils.input_utils import *
import robosuite as suite


def colorize_depth(depth, d_min=50, d_max=2000, d_scale=1000):
    depth = depth.copy() * d_scale
    depth = np.where(np.all((d_min < depth, depth < d_max), axis=0), depth, 0)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_8bit = cv2.convertScaleAbs(depth)
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return depth_colormap


def plot_peg_bps(peg_bps_obs):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(peg_bps_obs[:, 0], peg_bps_obs[:, 1], peg_bps_obs[:, 2], c='b', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal', 'box')
    fig.savefig('peg_bps.png')
    plt.close(fig)


def add_domain_randomize(env, camera=False, color=False, lighting=False):
    if any([camera, color, lighting]) is False:
        return env

    from robosuite.wrappers import DomainRandomizationWrapper
    import yaml
    print(f"Adding domain randomization: camera={camera}, color={color}, lighting={lighting}")

    randomize_cfg = yaml.safe_load(f"""
        camera:
            active: {'true' if camera else 'false'}
            pos_size: 0.05 # in m
            rot_size: 5 # in deg
            fov_size: 5.0
        color:
            active: {'true' if color else 'false'}
            local_color: false # Using a color similar to the original, controlled by local_rgb_interp.
            local_rgb_interp: 0.1 # [0, 1]
            local_material_interp: 0.2 # [0, 1]
        lighting:
            active: {'true' if lighting else 'false'}
    """)

    cam_cfg = randomize_cfg['camera']
    color_cfg = randomize_cfg['color']
    lighting_cfg = randomize_cfg['lighting']
    if color_cfg['active']:
        raise NotImplementedError(
            'Color randomization is not fixed yet.'
            'It does not work with our runtime peg and hole creation directly.'
        )
    COLOR_ARGS = {
        "geom_names": None,  # all geoms are randomized
        "randomize_local": color_cfg['local_color'],  # sample nearby colors
        "randomize_material": True,  # randomize material reflectance / shininess / specular
        "local_rgb_interpolation": color_cfg['local_rgb_interp'],
        "local_material_interpolation": color_cfg['local_material_interp'],
        "texture_variations": ["rgb", "checker", "noise", "gradient"],  # all texture variation types
        "randomize_skybox": True,  # by default, randomize skybox too
    }
    CAMERA_ARGS = {
        "camera_names": None,  # all cameras are randomized
        "randomize_position": cam_cfg['pos_size'] > 0,
        "randomize_rotation": cam_cfg['rot_size'] > 0,
        "randomize_fovy": cam_cfg['fov_size'] > 0,
        "position_perturbation_size": cam_cfg['pos_size'],
        "rotation_perturbation_size": cam_cfg['rot_size'] * (np.pi / 180),
        "fovy_perturbation_size": cam_cfg['fov_size'],
    }
    env = DomainRandomizationWrapper(
        env,
        randomize_color=color_cfg['active'],
        randomize_camera=cam_cfg['active'],
        randomize_lighting=lighting_cfg['active'],
        randomize_dynamics=False,  # error, need to be fixed
        randomize_on_reset=True,
        color_randomization_args=COLOR_ARGS,
        camera_randomization_args=CAMERA_ARGS,
        randomize_every_n_steps=10000,
        # NOTE: Do not assign the seed here. it will read the global seed,
        # resulting in a different seed for multi-environment setups
        # seed = None,
    )
    return env


def main(args):
    if args.tensorboard:
        tb_dir = Path('logs_kb_control') / f'{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir)

    action_dim = 6

    use_depth = True
    gripper_types = args.gripper

    # Load the desired controller
    arm_controller_config = suite.load_part_controller_config(default_controller=args.control)
    controller_configs = refactor_composite_controller_config(
        arm_controller_config, 'ur5e', ["right"]
    )

    obs_keys = [
        'robot0_non_priv_proprio-state',
        'robot0_priv_proprio-state',
        'cam_view_image',
        'cam_view_depth',
        'cam_view_segmentation_class',
    ]
    if args.use_peg_bps:
        print(f'[WARNING] peg bps is not supported, show the peg_pcd instead')
        # obs_keys.append('peg_bps_gt-state')
        obs_keys.append('peg_pcd-state')
    if args.shape_emb_src is not None:
        obs_keys.append('shape_emb-state')

    # initialize the task
    env = suite.make(
        env_name="SoftPegInHole",
        robots="UR5e",
        controller_configs=controller_configs,
        gripper_types=gripper_types,
        initialization_noise=None,
        has_renderer=True,
        ignore_done=True,
        has_offscreen_renderer=use_depth,
        camera_names="cam_view",
        use_camera_obs=use_depth,
        depth_mode=args.depth_mode,
        camera_depths=use_depth,
        camera_segmentations="class",  # {None, instance, class}
        camera_widths=640,
        camera_heights=480,
        use_proprio_names=[
            'wrist_pos_rel',
            'wrist_force',
            'wrist_torque',
            'peg_pos_rel',
            'peg_rot6d',
            'peg_alignment',
        ],
        use_peg_bps=args.use_peg_bps,
        shape_emb_src=args.shape_emb_src,
        horizon=1000,
        control_freq=20,
        deterministic_reset=False,
        success_reward=100,
        initial_pose=np.array([0.038, 0.665, 0.29, 1.0, 0.0, 0.0, 0.0]),
        shape=args.shape,
        shape_type=args.shape_type,
        hole_pos_var=args.hole_pos_var,
        peg_pos_var=args.peg_pos_var,
        peg_angle_var=args.peg_angle_var,
        peg_friction_range=[args.peg_friction, args.peg_friction],
        peg_size_range=args.peg_size,
        peg_mass_range=[args.peg_mass, args.peg_mass],
        peg_distance_weights=np.ones(3),
        # obs_pose_scale=0.1,
        # obs_force_scale=50.0,
        # obs_torque_scale=5.0,
        reward_type='baseline',
        camera_view_direction=args.view,
        peg_and_hole_color=dict(
            peg_default=[1.0, 1.0, 1.0],
            hole_default=[0.5, 0.5, 0.5],
            interp=0.2,
        ),
        translation_control_only=False,
        # force_termination_threshold=None,
        render_camera=None,
        renderer='mjviewer',
        renderer_config={'cam_config': {"lookat": [0.3, 0.5, 0.2],
                                        "distance": 0.2, "azimuth": 180, "elevation": -0, }}
    )
    env = add_domain_randomize(env, camera=args.randomize_camera, color=args.randomize_color, lighting=False)
    env = GymWrapper(env, flatten_obs=False, keys=obs_keys)

    env.set_curriculum(args.curriculum)
    env.reset()
    
    # For FDCC/COMPLIANCE controllers, explicitly reset goal to current pose to prevent auto-movement
    active_robot = env.robots[0]
    for arm in active_robot.arms:
        controller = active_robot.part_controllers[arm]
        if hasattr(controller, 'reset_goal'):
            controller.reset_goal()
    
    # env.viewer.set_camera(camera_id=0)
    # env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(env, pos_sensitivity=1.0, rot_sensitivity=1.0)

    # Wrap the keyboard on_press to capture initial pose at key press for debugging
    _orig_on_press = device.on_press
    device._debug_initial_pose = None  # (pos, quat)
    device._debug_last_key = None

    # Small helper to pretty-print keys from pynput (Key / KeyCode)
    def _key_to_str(key):
        try:
            # KeyCode with printable char
            if hasattr(key, 'char') and key.char is not None:
                return key.char
            # Special keys like Key.up / Key.left ...
            if hasattr(key, 'name') and key.name is not None:
                return key.name
            return str(key)
        except Exception:
            return str(key)

    def _debug_on_press(key):
        try:
            # Use the same controlled frame as the controller (grip_site), not gripper_eef_site (which has an offset)
            site_id = env.sim.model.site_name2id('gripper0_right_grip_site')
            init_pos = env.sim.data.site_xpos[site_id].copy()
            init_quat = mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3))
        except Exception:
            init_pos, init_quat = None, None
        device._debug_initial_pose = (init_pos, init_quat)
        device._debug_last_key = key
        # Immediate print at key press
        print(f"[KB DEBUG] press key={_key_to_str(key)} init_pos={init_pos} init_quat={init_quat}", flush=True)
        # Delegate to original handler
        _orig_on_press(key)

    # Rewire the pynput listener to use our debug on_press wrapper, so prints always show
    try:
        # Stop existing listener if running
        if hasattr(device, "listener") and device.listener is not None:
            try:
                device.listener.stop()
            except Exception:
                pass
        from pynput.keyboard import Listener
        device.listener = Listener(on_press=_debug_on_press, on_release=device.on_release)
        device.listener.start()
        print("[KB DEBUG] Rewired keyboard listener with debug on_press handler", flush=True)
    except Exception as e:
        print(f"[KB DEBUG] Failed to rewire listener: {e}", flush=True)

    device.start_control()

    # Start with gripper CLOSED by default (prevents dropping the peg)
    # Users can toggle with spacebar per the on-screen help
    for r_idx, robot in enumerate(env.robots):
        for a_idx, arm in enumerate(robot.arms):
            # Only set if the gripper is controllable (dof > 0)
            if robot.gripper[arm].dof > 0:
                device.grasp_states[r_idx][a_idx] = True

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
            controller = active_robot.part_controllers[arm]

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

            # For compliance controllers (FDCC, COMPLIANCE), append zero wrench to the 6D pose delta
            # Keyboard input only provides position/orientation commands, not force/torque
            if controller.name in ["FDCC", "COMPLIANCE"]:
                if controller.compliance_mode in ["variable_stiffness"]:
                    action_dict[arm] = np.concatenate([action_dict[arm], np.ones(6)*500])
                elif controller.compliance_mode in ["virtual_force"]:
                    action_dict[arm] = np.concatenate([action_dict[arm], np.zeros(6)])

        # Maintain gripper state for each robot but only update the active robot with action
        # Optionally zero yaw (az) rotation command for fairness testing (operate on per-arm action before vectorizing)
        if args.zero_az:
            for arm in active_robot.arms:
                if arm in action_dict and action_dict[arm] is not None and len(action_dict[arm]) >= 6:
                    action_dict[arm][5] = 0.0

        # (debug prints removed)

        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        obs, rew, terminated, truncated, info = env.step(env_action)

        # If a key was pressed this iteration, print initial and final EEF pose for debugging
        if getattr(device, "_debug_last_key", None) is not None:
            try:
                # Use the same controlled frame as the controller (grip_site), not gripper_eef_site (which has an offset)
                site_id = env.sim.model.site_name2id('gripper0_right_grip_site')
                final_pos = env.sim.data.site_xpos[site_id].copy()
                final_quat = mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3))
            except Exception:
                final_pos, final_quat = None, None

            init_pos, init_quat = (None, None)
            if getattr(device, "_debug_initial_pose", None) is not None:
                init_pos, init_quat = device._debug_initial_pose

            # Compute deltas when possible
            delta_pos = None
            if init_pos is not None and final_pos is not None:
                delta_pos = final_pos - init_pos

            print(f"[KB DEBUG] step key={_key_to_str(device._debug_last_key)} final_pos={final_pos} delta_pos={delta_pos}", flush=True)
            print(f"             final_quat={final_quat}", flush=True)

            # reset debug flags
            device._debug_last_key = None
            device._debug_initial_pose = None
        env.render()

        if not np.all(np.isclose(active_robot._joint_positions, prev_jpos, rtol=1e-5)):
            # print('joint position', active_robot._joint_positions)
            # print('hand position', active_robot._hand_pos)
            # non_priv_obs = obs["robot0_non_priv_proprio-state"]
            # priv_obs = obs["robot0_priv_proprio-state"]
            # print('non_priv')
            # for i, v in enumerate(non_priv_obs):
            #     print(f'\t{i}: {v:.6f}')
            # print('priv')
            # for i, v in enumerate(priv_obs):
            #     print(f'\t{i}: {v:.6f}')
            # print(f'{non_priv_obs.shape=} {non_priv_obs[:10]=}')
            # print(f'{non_priv_obs[:9]=}')
            # print(f'{non_priv_obs[9:9+5]=}')
            # if args.use_peg_bps:
            # print('peg_bps-state', obs['peg_bps-state'].shape)
            # print()
            pass
        prev_jpos = active_robot._joint_positions

        if args.visualize:
            img = obs['cam_view_image']
            depth = obs['cam_view_depth']
            seg = obs['cam_view_segmentation_class']
            seg_name2id = env.seg_name2id
            # UR5e, NullMount, Robotiq85GripperSoft,  PegObject, GuriguriLargeTriangleHoleObject

            mask_keys = ['PegObject', *[k for k in seg_name2id.keys() if 'hole' in k.lower()]]
            mask = np.any([seg == seg_name2id[k] for k in mask_keys], axis=0).squeeze()
            # cv2.imshow('mask', mask.astype(np.float32))

            # # rgb
            # img[~mask] = [0, 0, 0]
            # cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # cv2.moveWindow('image', 0, 0)
            # image_frame = cv2.getWindowImageRect('image')

            # # depth
            # # depth_color = colorize_depth(depth)
            # # cv2.imshow('depth_rgb', depth_color)
            depth[~mask] = 0
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            cv2.imshow('depth', depth)
            # cv2.moveWindow('depth', 0, image_frame[3])
            # print(depth.min(), depth.max(), depth.dtype)
            # saved_depth = (depth * 1000).astype('uint16')
            # print(saved_depth.min(), saved_depth.max(), saved_depth.dtype)
            # cv2.imwrite('depth.png', saved_depth)

            # seg
            # cv2.imshow('segmentation', seg.astype(np.float32) / seg.max())
            # colorized seg
            # seg = seg.squeeze()
            # seg_cnt = np.unique(seg)
            # class2clr = {k: np.around(k / 4 * 255).astype(np.uint8).repeat(3) for k in seg_cnt}
            # seg_color = np.zeros([*seg.shape, 3], dtype=np.uint8)
            # for k in seg_cnt:
            #     mask = (seg == k)
            #     seg_color[mask] = class2clr[k]
            # cv2.imshow('segmentation', seg_color)
            # cv2.moveWindow('segmentation', image_frame[2], 0)

            cv2.waitKey(1)
            # break

        if args.tensorboard:
            for k, v in obs.items():
                if 'cam_view' in k:
                    continue
                    # tb_writer.add_image(k, v, i)
                elif v.size == 1:
                    tb_writer.add_scalar(k, v, i)
                else:
                    tb_writer.add_scalars(k, {f'{j}': v[j] for j in range(len(v))}, i)

        if terminated or truncated:
            print('is_success', info['is_success'])
            print('is_truncated', info['is_truncated'])
            print("reset", f"{env.total_rewards=}")
            # input()
            obs, info = env.reset()
            active_robot = env.robots[0]
            # Re-initialize FDCC / compliance controller goals and keep gripper closed after reset
            for arm in active_robot.arms:
                controller = active_robot.part_controllers[arm]
                if hasattr(controller, 'reset_goal'):
                    controller.reset_goal()
            for r_idx, robot in enumerate(env.robots):
                for a_idx, arm in enumerate(robot.arms):
                    if robot.gripper[arm].dof > 0:
                        device.grasp_states[r_idx][a_idx] = True

        # print(env.eef_pos, env.eef_quat)
        peg_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_right_grip_site')]
        peg_quat = mat2quat(env.sim.data.site_xmat[env.sim.model.site_name2id(
            'gripper0_right_grip_site')].reshape(3, 3))
        eef_quat = env.eef_quat
        # print(peg_quat, eef_quat, quat_multiply(eef_quat, quat_inverse(peg_quat)))
        # print(f"{env.peg_pos_error} {peg_pos=}")

        # force = env.get_force_torque()
        # if i > 100:
        #     wrenchs.append(force)
        #     # print(f"{np.mean(wrenchs, axis=0)=}")
        #     print(f"{force[:3]=} {np.linalg.norm(force[:3])}")

        time.sleep(0.03)

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Map short CLI tokens to actual gripper classes (soft vs rigid)
    peg_dict = {
        '85-soft': 'Robotiq85GripperSoft',
        '85-rigid': 'Robotiq85Gripper',
        'hande-soft': 'RobotiqHandEGripperSoft',
        'hande-rigid': 'RobotiqHandEGripper',
    }

    parser = ArgumentParser()
    parser.add_argument('-g', '--gripper', default='85-soft', choices=peg_dict.keys(),
                        help='Select gripper type: 85-soft/85-rigid/hande-soft/hande-rigid')
    parser.add_argument('-c', '--control', default='OSC_POSE', choices=['OSC_POSITION', 'FDCC', 'COMPLIANCE', 'OSC_POSITION_CB'],
                        help='Controller to use for the arm')
    parser.add_argument('-s', '--shape', default=None)
    parser.add_argument('-st', '--shape_type', default='basic')
    parser.add_argument('-bps', '--use_peg_bps', action='store_true', help='use peg basis point set features')
    parser.add_argument('-se', '--shape_emb_src', type=str, default=None)
    parser.add_argument('-hpv', '--hole_pos_var', type=float, default=0.0)
    parser.add_argument('-ppv', '--peg_pos_var', type=float, default=0.0)
    parser.add_argument('-pav', '--peg_angle_var', type=float, default=0.0)
    parser.add_argument('-pf', '--peg_friction', type=float, default=1.0)
    parser.add_argument('-ps', '--peg_size', nargs='+', type=float, default=[1.0, 1.0])
    parser.add_argument('-pm', '--peg_mass', type=float, default=0.03)
    parser.add_argument('-view', '--view', type=str, default='left')
    parser.add_argument('-d', '--depth-mode', type=str, default='norm')
    parser.add_argument('-rcamera', '--randomize_camera', action='store_true', help='randomize camera')
    parser.add_argument('-rcolor', '--randomize_color', action='store_true', help='randomize color')
    parser.add_argument('-viz', '--visualize', action='store_true', help='visualize camera obs')
    parser.add_argument('-cl', '--curriculum', type=float, default=1.0, help='curriculum coefficient')
    parser.add_argument('-tb', '--tensorboard', action='store_true', help='log to tensorboard')
    parser.add_argument('--zero-az', action='store_true', help='Zero yaw (az) rotation command for fairness testing')
    parser.add_argument('-log', '--log_level', type=str, default='debug')
    args = parser.parse_args()
    args.gripper = peg_dict[args.gripper]

    logging.basicConfig(level=args.log_level.upper())
    main(args)
