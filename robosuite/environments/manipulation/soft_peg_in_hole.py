from collections import OrderedDict
import copy
import logging
from pathlib import Path

import numpy as np
import torch

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
    get_camera_pose, update_xml, compute_domain_randomization_range,
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
        table_full_size=(0.65, 0.65, 0.001),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="closeview",
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
        hard_reset_every_n_episodes=10,
        # TODO: refactor custom environment config
        hole_pos_var=None,
        peg_pos_var=None,
        peg_angle_var=None,
        peg_friction_range=[1., 1.],
        peg_size_range=[1., 1.],
        peg_mass_range=[0.03, 0.03],
        hole_max_z_height=0.10,
        wrist_max_z_height=0.10,  # relative to hole edge
        initial_pose=np.array([0.040, 0.662, 0.46, 1.0, 0.0, 0.0, 0.0]),
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
        force_termination_threshold=50.,
        peg_tilted_threshold=None,
        peg_distance_weights=np.array([1.0, 1.0, 10.0]),
        obs_pose_scale=1.0,
        obs_force_scale=1.0,
        obs_torque_scale=1.0,
        translation_control_only=True,
        peg_and_hole_color=None,
        spring_cfg=None,
        controller_kp_range=np.array([4.0, 4.5]),
        controller_kd_range=np.array([0.5, 1.5]),
    ):
        self.gripper_inertial_properties = None
        self.gripper_name = gripper_types

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
        self.PEG_Z_SIZE = 0.075
        self.HOLE_Z_SIZE = 0.140
        self.INSERT_Z_OFFSET = -0.03

        # curriculum learning
        self.curriculum_coef = 1.0
        self.curriculum_variance_coef = 1.0

        self.max_z_height = copy.copy(initial_pose[2])
        self.hole_min_z_height = self.table_offset[2] + (self.HOLE_Z_SIZE / 2)
        self.hole_max_z_height = self.hole_min_z_height + hole_max_z_height
        self.hole_pose = self.table_offset.copy()
        self.hole_pose[2] = self.hole_min_z_height
        self.wrist_max_z_height = wrist_max_z_height
        self.peg_to_wrist_pos = 0.22  # HARDCODED for now, updated in reset_internal

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
        self.spring_cfg = spring_cfg
        self.controller_kp_range = np.array(controller_kp_range)
        self.controller_kd_range = np.array(controller_kd_range)

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
        self.peg_tilted_threshold = peg_tilted_threshold
        self.peg_distance_weights = peg_distance_weights

        # TODO: do not hard code the condition
        self.reset_counter = 0
        self.hard_reset_every_n_episodes = hard_reset_every_n_episodes
        if self.user_defined_shape is None or self.peg_size_range[0] != self.peg_size_range[1]:
            hard_reset = True
            self.env_hard_reset = True
        else:
            hard_reset = False
            self.env_hard_reset = False

        self.ik = None

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
            curriculum_height_coef = min(x * 2.0, 1.0)  # from 0 to 0.5 increase height
            self.hole_pose[2] = self.hole_min_z_height
            # Always below the hole edge
            self.initial_pos[2] = self.hole_pose[2] + self.peg_to_wrist_pos + (self.INSERT_Z_OFFSET * (1 - curriculum_height_coef))
        else:
            # only after the peg is out of the hole, increase the variance of the hole pose and peg angle
            # interpolate the variance coef from 0.5 to 1.0
            self.curriculum_variance_coef = np.interp(x, (0.5, 1.0), (0.0, 1.0))
            hole_max_z_height = self.hole_min_z_height + (self.hole_max_z_height * self.curriculum_variance_coef)
            self.hole_pose[2] = np.random.uniform(low=self.hole_min_z_height, high=hole_max_z_height)
            # Always above the hole edge
            min_z_height = self.hole_pose[2] + self.peg_to_wrist_pos
            max_z_height = min_z_height + (self.wrist_max_z_height * self.curriculum_variance_coef)
            self.initial_pos[2] = np.random.uniform(low=min_z_height, high=max_z_height)

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        if self.reward_type == 'baseline':
            # progress reward
            progress_reward = (self.weighted_peg_dist_prev - self.weighted_peg_dist) / 0.001
            progress_reward = progress_reward
            # action smoothness reward
            action_smoothness_reward = - max(1.0, np.linalg.norm(action - self.action_prev) ** 2.0)
            step_reward = -0.1  # encourage early termination
            reward = progress_reward + action_smoothness_reward + step_reward
        elif self.reward_type == 'previous':
            # progress
            progress_reward = (self.weighted_peg_dist_prev - self.weighted_peg_dist) / 0.001
            # action
            PEG_ALIGNMENT_THRESHOLD = 0.005
            peg_pos, hole_pos = self.get_peg_and_hole_pos()
            peg_error = peg_pos - hole_pos
            is_aligned = np.linalg.norm(peg_error[:2]) < PEG_ALIGNMENT_THRESHOLD
            action_reward = - (action[2] ** 2) * (0.001 if is_aligned else 1.0)
            # action smooth
            action_smoothness_reward = - np.linalg.norm(action - self.action_prev) ** 2.0
            reward = progress_reward + action_reward + action_smoothness_reward
        else:
            raise ValueError(f'Invalid reward type {self.reward_type}')

        self.weighted_peg_dist_prev = self.weighted_peg_dist.copy()

        if self._check_success():
            reward = self.success_reward
        elif self._check_failure():
            reward = -self.success_reward

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
        self.hole = get_hole_object(self.shape, self.shape_type, self.hole_pose)
        self.peg = PegObject(self.shape, self.shape_type)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.hole],
            mujoco_objects_at_body={'gripper0_right_peg_wrapper': [self.peg]},
        )

        update_xml(
            xml_root=self.model.root,
            robot_configs=self.robot_configs,
            hole=self.hole,
            peg_size_range=self.peg_size_range,
            curriculum_variance_coef=self.curriculum_variance_coef,
            color_cfg=self.peg_and_hole_color,
            spring_cfg=self.spring_cfg
        )

        self.init_peg_pos = None
        self.init_xml = self.dump_xml("/root/robosim/model.xml")

    def dump_xml(self, filename=None):
        """
        Dumps the current MuJoCo model XML to a file or returns it as a string.

        Args:
            filename (str or None): If provided, writes the XML to this file path.
                                    If None, returns the XML string.
        Returns:
            str: If filename is None, returns the XML as a string. Otherwise, returns None.
        """
        # get mujoco XML as a string from the model
        import xml.etree.ElementTree as ET

        # Use robosuite's model or arena XML root if available
        xml_root = getattr(self, 'model', None)
        if xml_root is not None and hasattr(xml_root, 'root'):
            root = xml_root.root
        else:
            root = None

        if root is None:
            raise ValueError("No XML model root found to dump.")

        xml_str = ET.tostring(root, encoding="unicode")
        if filename is not None:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(xml_str)
            return None
        else:
            return xml_str

    @property
    def peg_pos_error(self):
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        return peg_pos - hole_pos

    @property
    def weighted_peg_dist(self):
        return np.linalg.norm(np.sqrt(self.peg_distance_weights) * self.peg_pos_error)

    def get_peg_to_wrist_pos(self):
        peg_pos = self.sim.data.get_site_xpos("gripper0_right_peg_ft_frame").copy()
        wrist_pos = self.sim.data.get_body_xpos("gripper0_right_gripper_base").copy()
        return abs(peg_pos[2] - wrist_pos[2])

    def get_peg_and_hole_pos(self):
        peg_pos = self.sim.data.get_site_xpos("gripper0_right_peg_ft_frame").copy()
        hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy() + np.array([0., 0., self.INSERT_Z_OFFSET])
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

    def wrist_pos_rel(self, obs_cache):
        wrist_pos = self.sim.data.get_body_xpos("gripper0_right_gripper_base")
        # We get the actual relative pose between the wrist and the hole
        # and use corrupter to handle the uncertainty in real-world
        _, hole_pos = self.get_peg_and_hole_pos()
        # Offset is the distance between the wrist and the hole
        # This is used to normalize the observation to the range [-1, 1]
        offset = np.array([0.0, 0.0, self.get_peg_to_wrist_pos()])
        return ((wrist_pos - hole_pos - offset) / self.obs_pose_scale).astype(np.float32)

    def wrist_force(self, obs_cache):
        wrist_force = self.get_force_torque()[:3]
        # Zero the force reading
        offset = np.array([0.0, 0.0, -1.92])
        return ((wrist_force + offset) / self.obs_force_scale).astype(np.float32)

    def wrist_torque(self, obs_cache):
        wrist_torque = self.get_force_torque()[3:]
        return (wrist_torque / self.obs_torque_scale).astype(np.float32)

    def wrist_rot6d_rel(self, obs_cache):
        # Compute the relative orientation from the initial rotation to current wrist rotation in 6D (ortho6) representation
        wrist_rot = self.sim.data.get_body_xmat("gripper0_right_gripper_base").reshape(3, 3)
        rel_rot = np.dot(np.linalg.inv(self.initial_rot), wrist_rot)
        return mat2ortho6(rel_rot).astype(np.float32)

    def peg_rot6d(self, obs_cache):
        return self.sim.data.get_site_xmat("gripper0_right_peg_ft_frame")[:, :2].flatten().astype(np.float32)

    def peg_pos_rel(self, obs_cache):
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        return ((peg_pos - hole_pos) / self.obs_pose_scale).astype(np.float32)

    def peg_alignment(self, obs_cache):
        PEG_ALIGNMENT_THRESHOLD = 0.005
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        peg_error = peg_pos - hole_pos
        aligned = np.linalg.norm(peg_error[:2]) < PEG_ALIGNMENT_THRESHOLD
        return aligned.astype(np.float32)

    def peg_vel(self, obs_cache):
        return self.sim.data.get_site_xvelp("gripper0_right_peg_ft_frame").copy().astype(np.float32)

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
        wrist_rot6d = self.wrist_rot6d_rel
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

        proprio_sensor_list = [
            [f"wrist_pos_rel", wrist_pos_rel, non_priv_modality],
            [f"wrist_rot6d", wrist_rot6d, non_priv_modality],
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

        # Only hard reset every N episodes to avoid resetting the environment too often
        if self.env_hard_reset and self.reset_counter % self.hard_reset_every_n_episodes == 0:
            self.hard_reset = True
        else:
            self.hard_reset = False

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

        if self.ik is None:
            self.ik = MuJoCoIKSolver(self.sim.model.get_xml(), [], "gripper0_right_ft_frame",
                                     joint_indexes=self.robots[0].joint_indexes,
                                     position_threshold=0.001,
                                     rotation_threshold=0.01,
                                     time_limit=0.1)

        result = self.ik.solve_ik(target_pos=init_wrist_pos,
                                  target_rot=quat2mat(self.initial_quat),
                                  initial_guess=init_qpos_guess)

        if result.success:
            self.robots[0].init_qpos = result.joint_angles
        else:
            log.warning(f"IK solution not found, using default init_qpos_guess. Error msg: {result.message}")
            self.robots[0].init_qpos = init_qpos_guess

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

        # Randomize the hole position
        hole_body_body = self.sim.model._model.body("hole_main")
        hole_body_body.pos = self.hole_pose

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

        self.peg_to_wrist_pos = self.get_peg_to_wrist_pos()
        self.reset_counter += 1

    def visualize(self, vis_settings):
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def _check_success(self):
        SUCCESS_THRESHOLD = 0.005
        peg_pos, hole_pos = self.get_peg_and_hole_pos()
        peg_error = peg_pos - hole_pos

        if np.linalg.norm(peg_error) < SUCCESS_THRESHOLD:
            return True
        return False

    def _check_failure(self):
        # FIXME: hardcoded value
        OUT_OF_PLAYGROUND_THRESHOLD = 0.14
        GOING_AWAY_FROM_GOAL_RATIO = 1.2
        # kinematic singularity termination
        is_singularity = np.linalg.det(self.robots[0].composite_controller.part_controllers['right'].J_full) < 0.01
        # Moving in the peg in the opposite direction to the goal
        is_going_away_from_goal = self.weighted_peg_dist > self.weighted_peg_dist_init * GOING_AWAY_FROM_GOAL_RATIO
        # Moving the wrist out of a safe zone even though the peg is stuck in the hole
        hole_pose = self.sim.data.body_xpos[self.hole_body_id][:2]  # ignore z
        is_out_of_playground = np.linalg.norm(self.eef_pos[:2] - hole_pose) > OUT_OF_PLAYGROUND_THRESHOLD
        # Contact force is to high, particularly between the wrist and the gripper (pushing down too hard)
        is_colliding = np.linalg.norm(self.get_force_torque()[:3]) > self.force_termination_threshold \
            if self.force_termination_threshold is not None else False
        # Peg is tilted too much, cause damage the spring
        is_peg_tilted = self.peg_to_straight_angle() > self.peg_tilted_threshold if self.peg_tilted_threshold is not None else False

        if is_singularity:
            return "singularity"
        elif is_colliding:
            return "collision"
        elif is_peg_tilted:
            return "peg_tilted"
        elif is_going_away_from_goal:
            return "going_away_from_goal"
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
        return self.robots[0].composite_controller.part_controllers['right'].eef_wrench

    def peg_to_straight_angle(self):
        nominal_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        peg_mat = self.sim.data.get_site_xmat("gripper0_right_peg_ft_frame").copy()
        angle_diff_rad = np.linalg.norm(quat2axisangle(mat2quat(peg_mat @ nominal_mat.T)))
        angle_diff_deg = np.rad2deg(angle_diff_rad)
        return angle_diff_deg.astype(np.float32)
