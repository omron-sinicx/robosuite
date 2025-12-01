from collections import OrderedDict
import copy
import logging
from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.controllers.parts.arm import fdcc, osc
from robosuite.utils.ik_solver import MuJoCoIKSolver
from robosuite.utils.observables import Observable, create_gaussian_noise_corrupter, create_uniform_sampled_delayer
from robosuite.utils.placement_samplers import CurriculumUniformRandomSampler
from robosuite.utils.transform_utils import *
from robosuite.environments.manipulation.env_utils import (
    get_peg_shape, get_hole_object,
    get_peg_point_cloud,
    get_camera_pose, setup_peg_and_hole, compute_domain_randomization_range,
    PegObject,
)

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)


class SoftPegInHole(ManipulationEnv):
    """
    This class corresponds to the peg in hole task for a single arm
    It is mostly copied and pasted from SoftPegInHole from Hai's robosuite repo (which was then mostly based off of Lift)
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise={"magnitude": 0.002, "type": "gaussian"},
        table_full_size=(0.65, 0.65, 0.025),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="sideview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=200,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        lite_physics=True,
        deterministic_reset=False,
        # TODO: refactor custom environment config
        hole_pos_var=None,
        peg_pos_var=None,
        peg_angle_var=None,
        peg_friction_range=[1., 1.],
        peg_size_range=[1., 1.],
        peg_mass_range=[0.03, 0.03],
        initial_pose=np.array([0.0, 0.65, 0.26, 1.0, 0.0, 0.0, 0.0]),
        camera_view_direction="left",
        use_proprio_names=None,
        depth_mode='norm',
        shape_emb_src=None,
        use_peg_bps=False,
        n_bps_pts=256,
        reward_type=None,
        success_reward=1.0,
        corrupted_obs={},
        delay_obs={},
        shape=None,
        shape_type='basic',
        force_termination_threshold=100.,
        peg_distance_weights=np.array([1.0, 1.0, 10.0]),
        obs_pose_scale=1.0,
        obs_force_scale=1.0,
        obs_torque_scale=1.0,
        translation_control_only=True,
        peg_and_hole_color=None,
        going_away_from_goal_threshold=1.2,
        out_of_playground_threshold=0.14,
    ):
        self.gripper_inertial_properties = None
        self.gripper_name = gripper_types
        self.going_away_from_goal_threshold = going_away_from_goal_threshold
        self.out_of_playground_threshold = out_of_playground_threshold

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # Place table center at the initial x,y pose of the robot
        self.table_offset = np.array((initial_pose[0], initial_pose[1], self.table_full_size[2]))
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        # object placement initializer
        self.placement_initializer = placement_initializer

        self.total_rewards = 0.0
        self.success_reward = success_reward

        # peg & hole shape
        assert shape_type, 'Must provide shape_type'
        self.user_defined_shape = shape
        self.shape = shape
        self.shape_type = shape_type

        # initial robot pose
        self.initial_pos = initial_pose[:3]
        self.initial_quat = initial_pose[3:]
        self.initial_rot = quat2mat(self.initial_quat)

        # Set default orientation for controller
        if translation_control_only:
            controller_configs['body_parts']['right']['default_orientation'] = self.initial_rot
        else:
            controller_configs['body_parts']['right']['default_orientation'] = None

        # goal settings
        INSERT_Z_OFFSET = -0.035
        PEG_Z_SIZE = 0.075
        HOLE_Z_SIZE = 0.14
        self.insertion_offset = np.array([0., 0., INSERT_Z_OFFSET])

        # curriculum learning
        self.curriculum_coef = 1.0
        self.curriculum_variance_coef = 1.0
        self.hole_edge_z_height = self.table_offset[2] + HOLE_Z_SIZE + PEG_Z_SIZE
        self.initial_z_height = 0.01 + self.hole_edge_z_height + self.insertion_offset[2]
        self.max_z_height = copy.copy(initial_pose[2])

        # env variance
        assert hole_pos_var is not None and hole_pos_var >= 0, f'Must provide hole_pos_var >= 0. Got {hole_pos_var}'
        assert peg_pos_var is not None and peg_pos_var >= 0, f'Must provide peg_pos_var >= 0. Got {peg_pos_var}'
        assert peg_angle_var is not None and peg_angle_var >= 0, f'Must provide peg_angle_var >= 0. Got {peg_angle_var}'
        assert len(peg_friction_range) == 2, f'peg_friction_range must have both min and max. Got {peg_friction_range}'
        assert len(peg_size_range) == 2, f'peg_size_range must have both min and max. Got {peg_size_range}'
        assert len(peg_mass_range) == 2, f'peg_mass_range must have both min and max. Got {peg_mass_range}'
        assert peg_friction_range[0] <= peg_friction_range[1], \
            f'peg_friction_range must have min <= max. Got {peg_friction_range}'
        assert 0 < peg_size_range[0] <= peg_size_range[1], \
            f'peg_size_range must have min > 0 and min <= max. Got {peg_size_range}'
        assert 0 < peg_mass_range[0] <= peg_mass_range[1], \
            f'peg_mass_range must have min > 0 and min <= max. Got {peg_mass_range}'
        self.hole_pos_var = float(hole_pos_var)
        self.peg_pos_var = float(peg_pos_var)
        self.peg_angle_var = float(peg_angle_var)
        self.peg_friction_range = np.array(peg_friction_range)
        self.peg_size_range = np.array(peg_size_range)
        self.peg_mass_range = np.array(peg_mass_range)
        self.peg_and_hole_color = peg_and_hole_color

        # observation
        assert use_proprio_names is not None and type(use_proprio_names) == list, \
            f'Must provide use_proprio_names in list. Got {use_proprio_names}'
        assert 0 < obs_pose_scale, f'Invalid pose scale {obs_pose_scale}'
        assert 0 < obs_force_scale, f'Invalid force scale {obs_force_scale}'
        assert 0 < obs_torque_scale, f'Invalid torque scale {obs_torque_scale}'
        self.use_proprio_names = use_proprio_names
        self.cam_view_direction = camera_view_direction
        self.depth_mode = depth_mode
        self.obs_pose_scale = obs_pose_scale
        self.obs_force_scale = obs_force_scale
        self.obs_torque_scale = obs_torque_scale
        self.use_peg_bps = use_peg_bps
        self.shape_emb_src = shape_emb_src

        if self.use_peg_bps:
            from bps_torch.bps import bps_torch  # basis point set
            # TODO: do not hard code the parameters
            FEATURE_TYPES = ['dists']  # ['dists', 'deltas', 'closest']
            BPS_TYPE = 'random_uniform'  # ['random_uniform', 'random_nonuniform', 'grid_cube', 'custom']

            self.bps_feature_types = FEATURE_TYPES
            self.bps_helper = bps_torch(bps_type=BPS_TYPE, n_bps_points=n_bps_pts, radius=1., n_dims=3,)

        # shape to files to load external files as observation
        train_eval = 'eval' if 'eval' in self.shape_type else 'train'
        if self.shape_emb_src is not None:
            if self.shape_emb_src.startswith('vqvae_'):
                name = f'{self.shape_emb_src[6:]}'  # skip vqvae_
                shape_emb_path = Path(__file__).parent.parent.parent.parent \
                    / 'third_parties/vqvae/results/' \
                    / f'{name}/{train_eval}.npy'
            elif self.shape_emb_src == 'uni3d':
                name = self.shape_type.replace('_', '-')
                shape_emb_path = Path(__file__).parent.parent.parent.parent \
                    / 'third_parties/Uni3D/results/' \
                    / f'{name}.npy'

            if not shape_emb_path.exists():
                raise FileNotFoundError(f"Shape embedding file not found: {shape_emb_path}")
            self.shape_emb_dict = np.load(shape_emb_path, allow_pickle=True).item()

        # observation corrupters
        self.corrupters = {}
        for k, v in corrupted_obs.items():
            if v is None:
                continue

            corrupted_mean, corrupted_std = v
            corrupted_mean = np.array(corrupted_mean)
            corrupted_std = np.array(corrupted_std)

            if 'pos_rel' in k:
                corrupted_mean /= obs_pose_scale
                corrupted_std /= obs_pose_scale
            elif 'force' in k:
                corrupted_mean /= obs_force_scale
                corrupted_std /= obs_force_scale
            elif 'torque' in k:
                corrupted_mean /= obs_torque_scale
                corrupted_std /= obs_torque_scale

            if 'align' in k:
                self.corrupters[k] = create_gaussian_noise_corrupter(
                    mean=corrupted_mean, std=corrupted_std, low=0.0, high=1.0
                )
            else:
                self.corrupters[k] = create_gaussian_noise_corrupter(
                    mean=corrupted_mean, std=corrupted_std
                )

        # observation delayer
        self.delayers = {}
        for k, v in delay_obs.items():
            if v is None:
                continue
            min_delay, max_delay = v
            self.delayers[k] = create_uniform_sampled_delayer(min_delay=min_delay, max_delay=max_delay)

        # reward
        assert reward_type, 'Must provide reward type'
        self.reward_type = reward_type
        self.force_termination_threshold = force_termination_threshold
        self.peg_distance_weights = peg_distance_weights

        # TODO: do not hard code the condition
        if self.user_defined_shape is None or self.peg_size_range[0] != self.peg_size_range[1]:
            hard_reset = True

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="NullMount",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            lite_physics=lite_physics,
        )

        # override base.py which hard codes it to be false, such that it can be set by the user
        # TODO: make it also toggle hole placement randomization
        self.deterministic_reset = deterministic_reset

    def set_curriculum(self, x):
        assert 0 <= x <= 1, "Curriculum must be between 0 and 1"
        self.curriculum_coef = x

        if x < 0.5:
            self.curriculum_variance_coef = 0.0  # No variations until the first phase of the curriculum is complete
            self.curriculum_height_coef = min(x * 2.0, 1.0)  # from 0 to 0.5 increase height
            self.initial_pos[2] = self.initial_z_height + \
                (self.hole_edge_z_height - self.initial_z_height) * self.curriculum_height_coef
        else:
            # only after the peg is out of the hole, increase the variance of the hole pose and peg angle
            # interpolate the variance coef from 0.5 to 1.0
            self.curriculum_variance_coef = np.interp(x, (0.5, 1.0), (0.0, 1.0))
            self.initial_pos[2] = np.random.uniform(low=self.hole_edge_z_height,
                                                    high=self.max_z_height)

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        if self.reward_type == 'baseline':
            # progress reward: only penalize moving away (sparse reward design)
            progress_reward = (self.weighted_peg_dist_prev - self.weighted_peg_dist) / 0.001
            # progress_reward = min(0.0, progress_reward)  # Only penalty for moving away
            progress_reward = max(0.0, progress_reward)
            # action smoothness reward
            action_smoothness_reward = - np.linalg.norm(action - self.action_prev) ** 2.0
            # force penalty - only apply when in contact (peg_pos_error_z < 0.02m, i.e., close to hole)
            peg_pos_error_z = self.peg_pos_error[2]  # Vertical error
            if peg_pos_error_z < 0.02:  # Only penalize force when very close to hole (contact phase)
                current_force = np.linalg.norm(self.get_force_torque()[:3])
                force_penalty = -0.005 * (current_force / 50.0) ** 2  # Light penalty only during insertion
            else:
                force_penalty = 0.0  # No penalty during approach phase
            step_reward = -0.1  # encourage early termination
            reward = progress_reward + action_smoothness_reward + force_penalty + step_reward
            self.weighted_peg_dist_prev = self.weighted_peg_dist.copy()
        else:
            raise ValueError(f'Invalid reward type {self.reward_type}')

        if self._check_success():
            # scale down the reward when using curriculum from 75% to 100%
            # reward += self.success_reward * min(self.curriculum_coef + 0.75, 1.0)
            reward += self.success_reward
        elif self._check_failure():
            reward += -self.success_reward

        self.total_rewards += reward

        return reward

    def _post_action(self, action):
        """
        Additional termination conditions compared to definition given in base.py
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        reward, self.done, _ = super()._post_action(action)

        # additional termination conditions compared to super
        is_success = self._check_success()
        failed_reason = self._check_failure()
        is_truncated = self.timestep >= self.horizon

        self.done = self.done or (failed_reason is not None) or is_success or is_truncated
        if self.ignore_done == True:
            self.done = False

        self.action_prev = action.copy()

        info = {
            'is_success': is_success,
            'is_truncated': is_truncated,
            'early_termination': failed_reason is not None,
            'early_termination_reason': failed_reason,
            'total_rewards': self.total_rewards,
            'timestep': self.timestep,
        }
        return reward, self.done, info

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        if self.use_camera_obs:
            assert len(self.camera_names) == 1, \
                f'Only support one camera, but get multiple {len(self.camera_names)} cameras'
            cam_pos, cam_quat = get_camera_pose(self.cam_view_direction, self.table_offset)
            self.cam_pos = cam_pos
            self.cam_quat_wxyz = cam_quat
            mujoco_arena.set_camera(
                camera_name=self.camera_names,
                pos=self.cam_pos,
                quat=self.cam_quat_wxyz,
                # doc: https://mujoco.readthedocs.io/en/latest/modeling.html#cameras
                camera_attribs=dict(
                    fovy=43.1456311,
                    # https://realsenseai.com/stereo-depth-cameras/stereo-depth-camera-d435/
                    ipd=0.050,
                ))

        # Check if shape_type is a peg set in the configuration file
        if self.user_defined_shape is None:
            self.shape = get_peg_shape(self.shape_type)
        else:
            self.shape = self.user_defined_shape

        if self.use_peg_bps:
            self.peg_pcd_canonical = get_peg_point_cloud(self.shape, self.shape_type)

        # Use the shape_type directly when creating the hole object
        # This ensures that when peg_shape is "custom", we use the dynamic hole generation
        self.hole = get_hole_object(self.shape, self.shape_type)
        self.peg = PegObject(self.shape, self.shape_type)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.hole)
        else:
            # NOTE: Randomizing the hole position can make the visualization misleading, suggesting the hole's position is unknown.
            # Apply the randomization based on the initial wrist position to keep the visualization accurate and the proprioception consistent.
            # hole_range_in_m = self.hole_pos_var * 0.001
            self.placement_initializer = CurriculumUniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.hole,
                # x_range=[-hole_range_in_m, hole_range_in_m],
                # y_range=[-hole_range_in_m, hole_range_in_m],
                x_range=[0, 0],
                y_range=[0, 0],
                # TODO: randomize rotation as well, but this I need to take observations relative to hole quaternions then
                rotation=0.0,
                # rotation=np.pi / 2.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.hole],
            mujoco_objects_at_body={'gripper0_right_peg_wrapper': [self.peg]},
        )

        # TODO: do this before the model is initialized
        setup_peg_and_hole(
            xml_root=self.model.root,
            robot_configs=self.robot_configs,
            hole=self.hole,
            shape=self.shape,
            shape_type=self.shape_type,
            peg_size_range=self.peg_size_range,
            curriculum_variance_coef=self.curriculum_variance_coef,
            color_cfg=self.peg_and_hole_color,
        )

        self.init_peg_pos = None

    @property
    def peg_pos_error(self):
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        return peg_pos - hole_pos

    @property
    def weighted_peg_dist(self):
        return np.linalg.norm(np.sqrt(self.peg_distance_weights) * self.peg_pos_error)

    def get_peg_and_hole_pos(self):
        peg_pos = self.sim.data.get_site_xpos("gripper0_right_peg_ft_frame").copy()
        hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy() + self.insertion_offset
        return peg_pos, hole_pos

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        # self.box_body_id = self.sim.model.body_name2id(self.box.root_body)

    def wrist_pos_rel(self, obs_cache):
        NORMALIZE_OFFSET = np.array([0.0, 0.0, -0.27])
        wrist_pos = self.sim.data.get_body_xpos("gripper0_right_gripper_base")
        # We get the actual relative pose between the wrist and the hole
        # and use corrupter to handle the uncertainty in real-world
        _, hole_pos = self.get_peg_and_hole_pos()
        # NOTE: [previous work] offset is a heuristic value that make the z-axis of the wrist approximately equal 0 when the insertion success
        offset = NORMALIZE_OFFSET
        # NOTE: [previous work] possible reasons: adjust the scale to make the NN easier to learn
        return ((wrist_pos - hole_pos + offset) / self.obs_pose_scale).astype(np.float32)

    def wrist_force(self, obs_cache):
        wrist_force = self.get_force_torque()[:3]
        # NOTE: [previous work] possible reasons: adjust the scale to make the NN easier to learn
        return (wrist_force / self.obs_force_scale).astype(np.float32)

    def wrist_torque(self, obs_cache):
        wrist_torque = self.get_force_torque()[3:]
        # NOTE: [previous work] possible reasons: adjust the scale to make the NN easier to learn
        return (wrist_torque / self.obs_torque_scale).astype(np.float32)

    def wrist_rot6d(self, obs_cache):
        return self.sim.data.get_body_xmat("gripper0_right_gripper_base")[:, :2].flatten().astype(np.float32)

    def peg_rot6d(self, obs_cache):
        return self.sim.data.get_site_xmat("gripper0_right_peg_ft_frame")[:, :2].flatten().astype(np.float32)

    def peg_pos_rel(self, obs_cache):
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        # NOTE: [previous work] possible reasons: adjust the scale to make the NN easier to learn
        return ((peg_pos - hole_pos) / self.obs_pose_scale).astype(np.float32)

    def peg_alignment(self, obs_cache):
        PEG_ALIGNMENT_THRESHOLD = 0.007
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        peg_error = peg_pos - hole_pos
        aligned = np.linalg.norm(peg_error[:2]) < PEG_ALIGNMENT_THRESHOLD
        return aligned.astype(np.float32)

    def peg_vel(self, obs_cache):
        return self.sim.data.get_site_xvelp("gripper0_right_peg_ft_frame").copy().astype(np.float32)

    def peg_to_straight_angle(self, obs_cache):
        nominal_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        peg_mat = self.sim.data.get_site_xmat("gripper0_right_peg_ft_frame").copy()
        angle_diff_rad = np.linalg.norm(quat2axisangle(mat2quat(peg_mat @ nominal_mat.T)))
        angle_diff_01 = angle_diff_rad / np.pi
        return angle_diff_01.astype(np.float32)

    def peg_hole_contact(self, obs_cache):
        raise NotImplementedError
        is_contacted = self.check_contact(self.peg_geom_names, self.hole_geom_names)
        return np.float32(is_contacted)

    def shape_emb(self, obs_cache):
        if self.shape_emb_src is None:
            ret = np.float32(0)
        elif self.shape_emb_src.startswith('vqvae_') or self.shape_emb_src == 'uni3d':
            if self.shape not in self.shape_emb_dict:
                raise ValueError(f"Shape embedding {self.shape} not found in shape embedding dictionary")
            ret = self.shape_emb_dict[self.shape].astype(np.float32)
        else:
            raise ValueError(f"Invalid shape embedding source: {self.shape_emb_src}")
        return ret

    def peg_pcd(self, obs_cache):
        if obs_cache.get('peg_pcd') is None:
            pcd_canonical = self.peg_pcd_canonical.copy()  # (n, 3)
            pcd_canonical = pcd_canonical - np.mean(pcd_canonical, axis=0)
            max_norm = np.max(np.linalg.norm(pcd_canonical, axis=1, keepdims=True))
            pcd_canonical = pcd_canonical / (max_norm if max_norm > 0 else 1.0)
            obs_cache['peg_pcd'] = pcd_canonical.astype(np.float32)
        return obs_cache['peg_pcd']

    def peg_bps_gt(self, obs_cache):
        import torch
        # TODO: how to determine this dim automatically?
        bps_feature = np.zeros([self.bps_helper.bps.shape[1],], dtype=np.float32)
        if self.use_peg_bps:
            if self.peg_pcd_canonical is None:
                raise ValueError("Peg point cloud is required for peg_bps")

            # TODO: encode from the pytorch3D.Meshes, rather than using manually stored point clouds
            # center the point cloud
            pcd_canonical = self.peg_pcd_canonical.copy()  # (n, 3)
            pcd_canonical = pcd_canonical - np.mean(pcd_canonical, axis=0)
            max_norm = np.max(np.linalg.norm(pcd_canonical, axis=1, keepdims=True))
            pcd_canonical = pcd_canonical / (max_norm if max_norm > 0 else 1.0)

            # Rotate the point cloud to the world frame
            peg_rot_mat = self.sim.data.get_site_xmat("gripper0_right_peg_ft_frame").astype(np.float32)  # (3, 3)
            pcd_world = np.dot(pcd_canonical, peg_rot_mat.T)  # (n, 3)

            # Compute the bps feature
            pcd_world_th = torch.from_numpy(pcd_world).unsqueeze(0)  # (1, n, 3)
            bps_feat_dict = self.bps_helper.encode(
                pcd_world_th, feature_type=self.bps_feature_types)  # {feat: (1, n, c)}
            assert self.bps_feature_types == ['dists'], \
                f'Only dists feature is supported for peg bps now. Got {self.bps_feature_types}'
            bps_feature = bps_feat_dict['dists'].cpu().squeeze(0).numpy()  # (n, )

        return bps_feature.astype(np.float32)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # overwrite parent setup observables call
        # the code below mirrors the code in robot/robots.py

        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        prefix = self.robots[0].robot_model.naming_prefix
        non_priv_modality = f"{prefix}non_priv_proprio"
        priv_modality = f"{prefix}priv_proprio"
        shape_modality = f"shape_emb"
        peg_bps_modality = f"peg_bps_gt"
        peg_pcd_modality = f"peg_pcd"

        # non_privileged modality
        wrist_pos_rel = self.wrist_pos_rel
        wrist_force = self.wrist_force
        wrist_torque = self.wrist_torque
        # privileged modality
        peg_pos_rel = self.peg_pos_rel
        peg_rot6d = self.peg_rot6d
        peg_alignment = self.peg_alignment
        peg_hole_contact = self.peg_hole_contact
        # peg_vel = self.peg_vel
        # peg_to_straight_angle = self.peg_to_straight_angle

        # shape modality
        shape_emb = self.shape_emb
        peg_bps_gt = self.peg_bps_gt
        peg_pcd = self.peg_pcd

        # @sensor(modality=modality)
        # def wrist_wrench(obs_cache):
        #     # NOTE self.sim.data.get_sensor doesn't seem to obtain the full sensor vector
        #     wrist_wrench = np.hstack(
        #         (
        #             self.sim.data._data.sensor("gripper0_force_ee").data,
        #             self.sim.data._data.sensor("gripper0_torque_ee").data,
        #         )
        #     )
        #     return wrist_wrench

        # @sensor(modality=modality)
        # def spring_angle(obs_cache):
        #     return np.array(
        #         [
        #             self.sim.data.get_joint_qpos("gripper0_flex_wrist_rx"),
        #             self.sim.data.get_joint_qpos("gripper0_flex_wrist_ry"),
        #             self.sim.data.get_joint_qpos("gripper0_flex_wrist_rz"),
        #         ]
        #     )

        # @sensor(modality=modality)
        # def peg_torque(obs_cache):
        #     return self.sim.data._data.sensor("gripper0_torque_peg").data

        # @sensor(modality=modality)
        # def joint_gains(obs_cache):
        #     # assume that all gains are common across joints
        #     assert np.linalg.norm(np.diff(self.robots[0].controller.kp)) < 10e-6
        #     assert np.linalg.norm(np.diff(self.robots[0].controller.kd)) < 10e-6

        #     raw_gains = np.array([self.robots[0].controller.kp[0], self.robots[0].controller.kd[0]])
        #     scale = np.array([5.0, 3.0])
        #     return np.log10(raw_gains) / scale

        # @sensor(modality=modality)
        # def hole_offset(obs_cache):
        #     hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
        #     hole_pos[2] -= 0.015

        #     hole_pos_nominal = np.array([0.0, 0.65, 0.15])
        #     scale = 0.01
        #     return (hole_pos - hole_pos_nominal) / scale

        # @sensor(modality=modality)
        # def peg_angle(obs_cache):
        #     return self.peg_angle / (5.0 * np.pi / 180.0)

        # @sensor(modality=modality)
        # def wrist_vel(obs_cache):
        #     return self.sim.data.get_body_xvelp("gripper0_gripper_base")

        # @sensor(modality=modality)
        # def peg_omega(obs_cache):
        #     return self.sim.data.get_site_xvelr("gripper0_right_peg_ft_frame")

        # @sensor(modality=modality)
        # def wrist_omega(obs_cache):
        #     return self.sim.data.get_body_xvelr("gripper0_gripper_base")

        # # time phase
        # # this isn't really proprioception, but whatever...
        # @sensor(modality=modality)
        # def time_phase(obs_cache):
        #     return np.array(
        #         [
        #             np.cos(2.0 * np.pi * self.timestep / self.horizon),
        #             np.sin(2.0 * np.pi * self.timestep / self.horizon),
        #         ]
        #     )

        proprio_sensor_list = [
            [f"wrist_pos_rel", wrist_pos_rel, non_priv_modality],
            [f"wrist_force", wrist_force, non_priv_modality],
            [f"wrist_torque", wrist_torque, non_priv_modality],
            [f"peg_pos_rel", peg_pos_rel, priv_modality],
            [f"peg_rot6d", peg_rot6d, priv_modality],
            [f"peg_alignment", peg_alignment, priv_modality],
        ]

        # Check if all self.corrupters are in the proprio_sensor_list
        corrupter_names = set(self.corrupters.keys())
        sensor_names = set([name for name, _, _ in proprio_sensor_list])
        assert corrupter_names.issubset(sensor_names), \
            f"Corrupter names {corrupter_names} not in sensor names {sensor_names}"

        # Create observables for this robot
        observables = OrderedDict()
        for name, s, m in proprio_sensor_list:
            active = name in self.use_proprio_names
            observables[name] = Observable(
                name=f'{prefix}{name}',
                sensor=s,
                sampling_rate=self.control_freq,
                modality=m,
                corrupter=self.corrupters.get(name),
                delayer=self.delayers.get(name),
                active=active,
            )

        if self.use_peg_bps:
            name = 'peg_bps_gt'
            observables[name] = Observable(
                name=name,
                sensor=peg_bps_gt,
                sampling_rate=self.control_freq,
                modality=peg_bps_modality,
                delayer=self.delayers.get(name),
                active=False
            )
            name = 'peg_pcd'
            observables[name] = Observable(
                name=name,
                sensor=peg_pcd,
                sampling_rate=self.control_freq,
                modality=peg_pcd_modality,
                delayer=self.delayers.get(name),
            )

        # add the shape embedding observation
        if self.shape_emb_src:
            name = 'shape_emb'
            observables[name] = Observable(
                name=name,
                sensor=shape_emb,
                sampling_rate=self.control_freq,
                modality=shape_modality,
            )

        # add the camera observation, follow RobotEnv
        if self.use_camera_obs:
            for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
                self.camera_names,
                self.camera_widths,
                self.camera_heights,
                self.camera_depths,
                self.camera_segmentations,
            ):

                # Add cameras associated to our arrays
                cam_sensors, cam_sensor_names, cam_sensor_modalities = self._create_camera_sensors(
                    cam_name, cam_w=cam_w, cam_h=cam_h, cam_d=cam_d, cam_segs=cam_segs, modality="image",
                    depth_mode=self.depth_mode
                )
                # Add camera obs
                for cam_name, cam_s, cam_modality in zip(cam_sensor_names, cam_sensors, cam_sensor_modalities):
                    observables[cam_name] = Observable(
                        name=cam_name,
                        sensor=cam_s,
                        sampling_rate=self.control_freq,
                        modality=cam_modality,
                        delayer=self.delayers.get(cam_name),
                    )

            # If any camera segmentation is not None, then we shrink all the sites as a hacky way to
            # prevent them from being rendered in the segmentation mask
            if not all(seg is None for seg in self.camera_segmentations):
                self.sim.model.site_size[:, :] = 1.0e-8

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """

        # hole position variance is equivalent to wrist position variance, also prevent misleading visualization
        init_wrist_pos = self.initial_pos.copy()
        init_wrist_pos[:2] += np.random.uniform(-self.hole_pos_var, self.hole_pos_var,
                                                2) * 0.001 * self.curriculum_variance_coef

        if self.robots[0].composite_controller is None or self.hard_reset:
            # instantiate controllers, only once
            super()._reset_internal()
            # Get gripper inertial properties for soft gripper (rigid grippers don't have right_gripper body)
            try:
                self.gripper_inertial_properties = self.sim.get_body_inertial_properties(f"gripper0_right_gripper_base")
            except ValueError:
                # Rigid gripper doesn't have the right_gripper body, set to None
                self.gripper_inertial_properties = None

        init_qpos_guess = np.array(
            [1.36314954, -1.21917949, 1.32688743, -1.67850362, -1.57077604, -1.77846293]
        )

        ik = MuJoCoIKSolver(self.sim.model.get_xml(), [], "gripper0_right_gripper_eef_site",
                            joint_indexes=self.robots[0].joint_indexes,
                            position_threshold=0.001,
                            rotation_threshold=0.01,
                            time_limit=0.1)
        result = ik.solve_ik(target_pos=init_wrist_pos,
                             target_rot=quat2mat(self.initial_quat),
                             initial_guess=init_qpos_guess)

        if result.success:
            self.robots[0].init_qpos = result.joint_angles
        else:
            log.warning(f"IK solution not found, using default init_qpos_guess. Error msg: {result.message}")
            self.robots[0].init_qpos = init_qpos_guess

        # Reset all object positions using initializer sampler
        # if not self.deterministic_reset:
        # Sample from the placement initializer for all objects
        object_placements = self.placement_initializer.sample()

        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Randomize the peg angle
        peg_wrapper_body = self.sim.model._model.body("gripper0_right_peg_wrapper")
        if self.init_peg_pos is None:
            self.init_peg_pos = peg_wrapper_body.pos.copy()

        self.peg_pos_offset = self.peg_pos_var * 0.001 * np.random.uniform(-1.0, 1.0, 3) * self.curriculum_variance_coef
        self.peg_angle = (self.peg_angle_var * np.pi / 180.0) * \
            (np.random.uniform(-1.0, 1.0) * self.curriculum_variance_coef)
        peg_quat_xyzw = axisangle2quat(np.array([0, self.peg_angle, 0]))
        peg_quat_wxyz = convert_quat(peg_quat_xyzw, to="wxyz")
        # mujoco use the wxyz quaternion, so initialize the rotation with (1,0,0,0)
        peg_wrapper_body.quat = quat_multiply(np.array([1.0, 0.0, 0.0, 0.0]), peg_quat_wxyz)
        peg_wrapper_body.pos = self.init_peg_pos + self.peg_pos_offset

        # Randomize the peg mass
        min_peg_mass, max_peg_mass = compute_domain_randomization_range(
            'peg_mass', self.peg_mass_range[0], self.peg_mass_range[1], self.curriculum_coef)
        new_mass = np.random.uniform(min_peg_mass, max_peg_mass)
        peg_wrapper_body.mass = new_mass

        # Randomize the peg friction
        peg_body_name = 'peg_main'
        peg_body_id = self.sim.model.body_name2id(peg_body_name)
        min_peg_friction, max_peg_friction = compute_domain_randomization_range(
            'peg_friction', self.peg_friction_range[0], self.peg_friction_range[1], self.curriculum_coef)
        new_friction = [np.random.uniform(min_peg_friction, max_peg_friction), 0.005, 0.0001]
        for geom_id in range(self.sim.model.ngeom):
            if self.sim.model.geom_bodyid[geom_id] == peg_body_id:
                self.sim.model.geom_friction[geom_id] = new_friction

        # Adjust the camera orientation to face the hole
        # if self.use_camera_obs:
        #     cam_id = self.sim.model.camera_name2id(self.camera_names[0])
        #     cam = self.sim.model.camera(cam_id)

        #     Adjust the camera orientation to face the hole
        #     FIXME: a weird way to get the hole position in the world frame
        #     hole_pos = self.sim.data.get_joint_qpos("hole_joint0")[:3]
        #     cam_quat = quat_from_vec1_to_vec2(np.array([0, 0, 1]), cam_pos - hole_pos)
        #     cam_quat = convert_quat(cam_quat, to="wxyz")
        #     cam_quat = quat_multiply(np.array([-np.sqrt(2), 0, 0, np.sqrt(2)]), cam_quat)

        #     cam.quat = cam_quat
        #     self.cam_fovy = self.sim.model.cam_fovy[cam_id]

        super()._reset_internal()  # reset controllers

        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        self.action_prev = np.zeros(self.action_dim)
        self.peg_pos_init = peg_pos
        self.weighted_peg_dist_prev = self.weighted_peg_dist.copy()
        self.weighted_peg_dist_init = self.weighted_peg_dist.copy()
        self.peg_vel_prev = np.zeros(3)

        self.total_rewards = 0.0

        controller = self.robots[0].composite_controller.part_controllers['right']

        if self.gripper_name == "Robotiq85GripperSoft":
            # Only set kp and kd for the soft gripper
            controller.kp = np.ones(6) * 10.0 ** np.random.uniform(4.0, 4.5)
            controller.kd = np.sqrt(controller.kp) * np.random.uniform(0.5, 1.5)

    def visualize(self, vis_settings):
        """
        TODO
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # TODO: additional visualization

    def _check_success(self):
        SUCCESS_THRESHOLD = 0.005
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        peg_error = peg_pos - hole_pos

        if np.linalg.norm(peg_error) < SUCCESS_THRESHOLD:
            return True
        return False

    def _check_failure(self):
        # kinematic singularity termination
        is_singularity = np.linalg.det(self.robots[0].composite_controller.part_controllers['right'].J_full) < 0.01
        # Moving in the peg in the opposite direction to the goal
        # is_going_away_from_goal = self.weighted_peg_dist > self.weighted_peg_dist_init * self.going_away_from_goal_threshold
        is_going_away_from_goal = False
        # Moving the wrist out of a safe zone even though the peg is stuck in the hole
        hole_pose = self.sim.data.body_xpos[self.hole_body_id][:2]  # ignore z
        is_out_of_playground = np.linalg.norm(self.eef_pos[:2] - hole_pose) > self.out_of_playground_threshold
        # Contact force is to high, particularly between the wrist and the gripper (pushing down too hard)
        is_colliding = np.linalg.norm(self.get_force_torque()[:3]) > self.force_termination_threshold \
            if self.force_termination_threshold is not None else False

        if is_singularity:
            return "singularity"
        elif is_going_away_from_goal:
            return "going_away_from_goal"
        elif is_colliding:
            return "collision"
        elif is_out_of_playground:
            return "out_of_playground"
        return None

    @property
    def eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id['right']])

    @property
    def eef_quat(self):
        return mat2quat(self.eef_rot)

    @property
    def eef_rot(self):
        return self.sim.data.site_xmat[self.robots[0].eef_site_id['right']].reshape(3, 3)

    def get_force_torque(self):
        ctrl = self.robots[0].composite_controller.part_controllers['right']
        wrench_props = ["current_wrench", "base_wrench", "eef_wrench", "world_wrench"]
        for prop in wrench_props:
            if hasattr(ctrl, prop):
                attr = getattr(ctrl, prop)
                try:
                    wrench = attr() if callable(attr) else attr
                    if wrench is not None:
                        # Check for correct type and shape (should be array-like, length 6)
                        arr = np.asarray(wrench)
                        if arr.shape == (6,):
                            return arr
                except Exception:
                    continue
        raise AttributeError("Controller does not provide a valid 6D wrench property (current_wrench, base_wrench, eef_wrench, or world_wrench)")
        