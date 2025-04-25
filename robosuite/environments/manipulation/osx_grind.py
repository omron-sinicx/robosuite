import logging
import numpy as np
from collections import OrderedDict
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MortarObject, CylinderObject
from robosuite.models.objects.xml_objects import MortarSDFObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.ik_solver import MuJoCoIKSolver
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.traj_utils import compute_max_step_size, generate_mortar_trajectory
from robosuite.controllers.parts.arm.fdcc import ForwardDynamicsComplianceController
import robosuite.utils.transform_utils as T


# Default Grind environment configuration
DEFAULT_GRIND_CONFIG = {
    # settings for reward
    "reward_weights": {
        "tracking_trajectory_error": 1.0,  # reward for following the trajectory reference
        "tracking_force_error": 1.0,  # reward for pushing into the mortar according to te force reference
        "action_smoothness": 0.1,  # reward for smooth actions (low value to start)
        "speed": 1.0,  # reward for speed
    },
    "task_complete_reward": 10.0,  # reward per task done
    "early_termination_penalty": -10.0,  # penalty for early termination
    "waypoint_completion_delay": 0.0,  # delay in seconds for waypoint completion
    "step_penalty": -1,  # penalty for each step

    "force_torque_normalization": [50.0, 50.0, 50.0, 10.0, 10.0, 10.0],  # max load (N)
    "pose_normalization": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # max distance between waypoints

    "table_full_size": [0.8, 0.8, 0.05],
    "table_friction": [1.0, 5e-3, 1e-4],

    # settings for thresholds
    "force_torque_limits": [50.0, 50.0, 50.0, 10.0, 10.0, 10.0],  # maximum eef force/torque allowed (N | N/m)

    # tracking settings
    "tracking_trajectory_threshold": 0.005,
    "tracking_force_threshold": 1.0,
    "tracking_trajectory_method": 'per_error_threshold',

    "reset_with_ik": True,

    # Trajectory settings
    "randomize_reference_trajectory": False,
    "num_waypoints": 1000,
    "duration": 10,
    "duration_range": [3.0, 30.0],
    "target_force": 10.0,  # N
    "target_force_range": [1.0, 15.0],

    # action settings
    "action_type": "stiffness_kp",  # "stiffness_kp" or "virtual_force" or "combined" or "none"
    "action_ndim": 4,  # 4D or 12D
    "action_change_type": "immediate",  # "immediate" or "progressive"
    "action_step_size": 0.01,  # step size for progressive action change

    # misc settings
    "early_terminations": True,  # Whether we allow for early terminations or not
    "clip_reward": True,  # Whether we clip the reward or not
    "relative_wrench_mode": "controlled_directions_only",  # "controlled_directions_only" or "all"

    # Task settings
    # Mortar parameters
    "mortar_height": 0.047,  # (m)
    "mortar_max_radius": 0.04,  # (m)
    "mortar_mode": "mesh",  # "SDA" or "mesh" Convex Decomposition Approximation
    "spawn_mortar": True,
    "mortar_diameter": 0.08,  # diameter of the mortar (m)
    "mortar_inner_height": 0.012,  # height of the mortar inner surface (m)
    "desired_height": 0.005,  # desired grinding height (m)
    "max_inclination_angle": 0.5,  # fraction of mortar radius for inclination
    "initial_orientation": [0.0, 1.0, 0.0, 0.0],  # initial quaternion orientation
    "initial_position": [0, 0, 0.8],  # initial position offset
    "mortar_space_threshold_max": 0.1,  # maximum distance from the mortar the eef is allowed to diverge (m)
}


TRACKING_METHODS = [
    'per_step',  # update waypoints after every step
    'per_error_threshold'  # update waypoints after the tracking error is smaller than `tracking_trajectory_threshold`
]


def scale_action(action, input_min, input_max, output_min, output_max):
    action_scale = abs(output_max - output_min) / abs(input_max - input_min)
    action_output_transform = (output_max + output_min) / 2.0
    action_input_transform = (input_max + input_min) / 2.0
    action = np.clip(action, input_min, input_max)
    transformed_action = (action - action_input_transform) * action_scale + action_output_transform

    return transformed_action


class OSXGrind(ManipulationEnv):
    """
    This class corresponds to the grinding task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. For this environment, setting a value other than the default ("Grinder")
            will raise an AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (mortar) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
            parameters, see the default configuration dict at the top of this file.
            If None is specified, the default configuration will be used.

    Raises:
        AssertionError: [Gripper specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="Grinder",
        initialization_noise="default",
        use_camera_obs=True,
        reward_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        task_config=DEFAULT_GRIND_CONFIG,
        renderer="mjviewer",
        renderer_config=None,
        reference_trajectory=None,
        enable_logging=True
    ):
        if not enable_logging:
            logging.getLogger().setLevel(logging.ERROR)
        # Assert that the gripper type is Grinder
        assert (
            gripper_types == "Grinder"
        ), "Tried to specify gripper other than Grinder in Grind environment!"

        self.horizon = horizon
        self.task_config = task_config

        self.early_terminations = self.task_config["early_terminations"]
        self.waypoint_completion_delay = self.task_config["waypoint_completion_delay"]
        self.force_torque_normalization = np.array(self.task_config["force_torque_normalization"])
        self.pose_normalization = np.array(self.task_config["pose_normalization"])

        self.force_torque_limits = self.task_config['force_torque_limits']

        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_weights = self.task_config['reward_weights']
        self.clip_reward = self.task_config["clip_reward"]
        self.task_complete_reward = self.task_config["task_complete_reward"]
        self.early_termination_penalty = self.task_config["early_termination_penalty"]
        self.step_penalty = self.task_config["step_penalty"]
        # settings for table top and task space
        self.mortar_height = self.task_config["mortar_height"]
        self.mortar_radius = self.task_config["mortar_max_radius"]
        self.mortar_space_threshold_max = self.task_config["mortar_space_threshold_max"]
        self.mortar_mode = self.task_config["mortar_mode"]
        self.spawn_mortar = self.task_config["spawn_mortar"]

        self.reset_with_ik = self.task_config["reset_with_ik"]

        # settings for table top
        self.table_full_size = self.task_config["table_full_size"]
        self.table_offset = np.array([0, 0, 0.8])
        self.table_friction = self.task_config["table_friction"]
        self.task_box = np.array([self.mortar_radius, self.mortar_radius, self.mortar_height+self.table_offset[2]]) + self.mortar_space_threshold_max

        # references to follow
        self.current_waypoint_index = 0
        self.target_force = self.task_config["target_force"]
        self.target_force_range = self.task_config["target_force_range"]

        self.ft_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.duration = self.task_config["duration"]
        self.duration_range = self.task_config["duration_range"]
        self.num_waypoints = control_freq * self.duration // 10
        self.seconds_per_waypoint = 10 / control_freq
        self.step_duration = max(1.0/500, self.duration / self.num_waypoints)  # Minimum 500Hz like in real UR5e
        self.last_step_time = 0

        self.tracking_trajectory_method = self.task_config['tracking_trajectory_method']
        self.tracking_trajectory_threshold = np.array(self.task_config['tracking_trajectory_threshold'])
        self.tracking_force_threshold = np.array(self.task_config['tracking_force_threshold'])
        # Verify the proposed impedance mode is supported
        assert self.tracking_trajectory_method in TRACKING_METHODS, (
            "Error: unsupported tracking method"
            "Inputted tracking method: {}, Supported methods: {}".format(self.tracking_trajectory_method, TRACKING_METHODS)
        )

        self.controller_type = np.array(controller_configs['body_parts']['right']['type'])
        self.position_control_dims = np.array(controller_configs['body_parts']['right']['selection_matrix'])
        self.force_control_dims = np.ones(6) - self.position_control_dims

        # Add an extra waypoint to make sure that every waypoint is
        # tracked before considering the tracking completed
        self.randomize_reference_trajectory = self.task_config['randomize_reference_trajectory']
        if reference_trajectory is None:
            self.reference_trajectory = self._randomize_reference_trajectory(control_freq)
        else:
            self.reference_trajectory = reference_trajectory

        if not self.randomize_reference_trajectory:
            self.reference_force = np.array([[0, 0, self.target_force, 0, 0, 0]] * self.num_waypoints)

        # actor action subset
        self.action_ndim = self.task_config['action_ndim']
        self.action_type = self.task_config['action_type']
        self.action_change_type = self.task_config['action_change_type']
        self.action_step_size = self.task_config['action_step_size']
        self.previous_action = np.zeros(self.action_ndim)
        self.current_action = np.zeros(self.action_ndim)

        self.placement_initializer = None

        self.input_min = np.array([-1]*6)
        self.input_max = np.array([1]*6)

        self.reward_dict = {
            "force_reward": 0.0,
            "traj_reward": 0.0,
            "action_smoothness": 0.0,
            "step_penalty": 0.0,
            "force_total_reward": 0.0,
            "traj_total_reward": 0.0,
            "speed_reward": 0.0,
            "speed_total_reward": 0.0,
            "action_smoothness_total_reward": 0.0,
        }

        self.init_qpos = np.array(
            [-0.24163013, -0.88630004,  1.99429391, -2.6787902, -1.57079633, -4.95401911]
        )

        self.cumulative_reward = 0.0
        self.global_timestep = 0
        self.ik = None
        self.translated_action = OrderedDict()
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
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
            lite_physics=lite_physics,
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
        )

    def compute_cartesian_compliance_controller_targets(self, action):
        controller: ForwardDynamicsComplianceController = self.robots[0].composite_controller.part_controllers['right']
        controller_targets = np.concatenate([
            self.reference_trajectory[self.current_waypoint_index],
            self.reference_force[self.current_waypoint_index]
        ])

        if self.action_change_type == "progressive":
            action = self.previous_action + (self.action_step_size * action)

        # change controller params
        if self.action_type is None:
            pass
        elif self.action_type == "stiffness_kp":

            if self.action_ndim == 4:
                # Reconstruct the action to be 12D
                action_kp = np.concatenate([[action[0]]*3, [action[1]]*3])
                action_stiffness = np.concatenate([[action[2]]*3, [action[3]]*3])
                action_kp = scale_action(action_kp, self.input_min, self.input_max,
                                         controller.kp_min, controller.kp_max)
                action_stiffness = scale_action(action_stiffness, self.input_min, self.input_max,
                                                controller.stiffness_min, controller.stiffness_max)
            elif self.action_ndim == 12:
                action_kp = scale_action(action[:6], self.input_min, self.input_max,
                                         controller.kp_min, controller.kp_max)
                action_stiffness = scale_action(action[6:], self.input_min, self.input_max,
                                                controller.stiffness_min, controller.stiffness_max)
            else:
                raise ValueError(f"Unsupported action dimension: {self.action_ndim}. Only 4 or 12 are supported.")

            controller.kp = action_kp
            controller.stiffness = action_stiffness
            # controller.kd = action_kp * controller.damping_ratio
            self.translated_action = OrderedDict({
                "kp": [action_kp[0], action_kp[3]],
                "stiffness": [action_stiffness[0], action_stiffness[3]],
            })

        elif self.action_type == "virtual_force":
            assert action.shape == (6,), f"Invalid action shape: {action.shape} != (6,)"
            controller.virtual_force = scale_action(action, self.input_min, self.input_max,
                                                    controller.virtual_force_min, controller.virtual_force_max)
            self.translated_action = OrderedDict({
                "virtual_force": controller.virtual_force,
            })

        elif self.action_type == "combined":
            assert action.shape == (10,), f"Invalid action shape: {action.shape} != (10,)"
            # Reconstruct the action to be 12D
            action_kp = np.concatenate([[action[0]]*3, [action[1]]*3])
            action_stiffness = np.concatenate([[action[2]]*3, [action[3]]*3])
            action_kp = scale_action(action_kp, self.input_min, self.input_max,
                                     controller.kp_min, controller.kp_max)
            action_stiffness = scale_action(action_stiffness, self.input_min, self.input_max,
                                            controller.stiffness_min, controller.stiffness_max)
            controller.kp = action_kp
            controller.stiffness = action_stiffness
            # controller.kd = action_kp * controller.damping_ratio
            controller.virtual_force = scale_action(action[4:], self.input_min, self.input_max,
                                                    controller.virtual_force_min, controller.virtual_force_max)
            self.translated_action = OrderedDict({
                "kp": [action_kp[0], action_kp[3]],
                "stiffness": [action_stiffness[0], action_stiffness[3]],
                "virtual_force": controller.virtual_force,
            })
        elif self.action_type == "cartesian_pose":
            controller_targets[:7] = action
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}. Only 'stiffness_kp', 'virtual_force', and 'combined' are supported.")

        self.action_data = {
            "kp": controller.kp,
            "kd": controller.kd,
            "stiffness": controller.stiffness,
            "virtual_force": controller.virtual_force,
        }

        return controller_targets

    def step(self, policy_action):
        """
        Take a step in the environment with the given action.

        Args:
            action (np.array): Action array that can be either 4D or 12D:
                - 4D: [kp_pos, kp_ori, stiffness_pos, stiffness_ori] where each value is replicated 3 times
                - 12D: [kp_pos_x, kp_pos_y, kp_pos_z, kp_ori_x, kp_ori_y, kp_ori_z,
                        stiffness_pos_x, stiffness_pos_y, stiffness_pos_z,
                        stiffness_ori_x, stiffness_ori_y, stiffness_ori_z]
                Values should be in range [-1, 1] and will be scaled to controller limits.

        Returns:
            4-tuple:
                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) whether the episode has ended
                - (dict) info about current episode state

        Raises:
            ValueError: If action_ndim is not 4, 6, or 12
        """
        action = policy_action.copy()
        assert action.shape == (self.action_ndim,), f"Invalid action shape: {action.shape} != {self.action_ndim}"

        self._update_waypoint_index(action)

        if self.controller_type == "FDCC":
            controller_targets = self.compute_cartesian_compliance_controller_targets(action)
        elif self.controller_type == "JOINT_VELOCITY":
            controller_targets = action
        else:
            raise ValueError(f"Unsupported controller type: {self.controller_type}. Only 'FDCC' and 'JOINT_VELOCITY' are supported.")

        return super().step(controller_targets)

    def reward(self, action=None):

        reward = 0.0

        terminated, reason = self._check_terminated()
        if terminated:
            if reason == "TRACKING COMPLETED":
                return self.task_complete_reward
            else:
                return self.early_termination_penalty

        # Reward for pushing into mortar with desired linear forces
        distance_from_ref_force = -self.tracking_force_error
        force_reward = self.reward_weights['tracking_force_error'] * distance_from_ref_force

        # Reward for following desired linear trajectory
        tracking_trajectory_error = -self.tracking_error
        traj_reward = self.reward_weights['tracking_trajectory_error'] * tracking_trajectory_error

        # Reward for smooth actions - penalize squared differences between consecutive actions
        if self.current_action is not None and hasattr(self, 'previous_action') and self.action_change_type == "immediate":
            action_smoothness_penalty = -self.reward_weights['action_smoothness'] * np.sum((self.current_action - self.previous_action)**2)
        else:
            action_smoothness_penalty = 0.0

        speed_reward = self.reward_weights['speed'] * (self.current_waypoint_index - self.global_timestep) / self.num_waypoints

        reward = force_reward + traj_reward + action_smoothness_penalty + self.step_penalty + speed_reward
        # print(f"{force_reward=:0.02f} {traj_reward=:0.02f} {action_smoothness_penalty=:0.02f} {self.step_penalty=:0.02f} {speed_reward=:0.02f}")

        if self.clip_reward:
            reward = np.clip(reward, -2.0, 1.0)

        self.reward_dict["force_reward"] = force_reward
        self.reward_dict["traj_reward"] = traj_reward
        self.reward_dict["action_smoothness"] = action_smoothness_penalty
        self.reward_dict["step_penalty"] = self.step_penalty
        self.reward_dict["speed_reward"] = speed_reward
        self.reward_dict["force_total_reward"] += force_reward
        self.reward_dict["traj_total_reward"] += traj_reward
        self.reward_dict["speed_total_reward"] += speed_reward
        self.reward_dict["action_smoothness_total_reward"] += action_smoothness_penalty
        # print(f"{force_reward=} {traj_reward=} {self.step_penalty=}")

        return reward

    def _compute_relative_distance(self):
        relative_distance = T.compute_pose_error(self.reference_trajectory[self.current_waypoint_index], self.eef_pose)

        # track error of the actions controlled by the policy
        # normalize by the trajectory follow normalization
        normalized_relative_distance = relative_distance / self.max_step_size

        # only consider the error for the position controlled directions
        self.tracking_error = np.linalg.norm(normalized_relative_distance * self.position_control_dims)
        return normalized_relative_distance

    def _compute_relative_wrenches(self):
        # in base frame
        relative_wrench = self.reference_force[self.current_waypoint_index] - self.eef_wrench
        # normalize by the force follow normalization
        normalized_relative_wrench = relative_wrench / self.force_torque_normalization

        # only consider the error for the force controlled directions
        tracking_force_error = normalized_relative_wrench * self.force_control_dims
        self.tracking_force_error = np.linalg.norm(tracking_force_error)

        # Only return values where (1-selection_matrix) equals 1 (force-controlled directions)
        if self.task_config["relative_wrench_mode"] == "controlled_directions_only":
            force_controlled_indices = np.where(self.force_control_dims == 1)[0]
            force_controlled_values = normalized_relative_wrench[force_controlled_indices]
            return force_controlled_values
        elif self.task_config["relative_wrench_mode"] == "all":
            return normalized_relative_wrench
        else:
            raise ValueError(f"Unsupported relative_wrench_mode: {self.task_config['relative_wrench_mode']}, only supported modes are 'controlled_directions_only' and 'all'")

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        self.robots[0].init_qpos = np.array([-0.24317403, -0.82343785,  1.99487586, -2.74223148, -1.57079607,  1.32762232])

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        if self.mortar_mode == "mesh":
            self.mortar = MortarObject(
                name="mortar",
            )
        elif self.mortar_mode == "SDF":
            self.mortar = MortarSDFObject(
                name="mortar",
                height=0.0,
                radius=0.045,
                thickness=0.005
            )
        else:
            raise ValueError(f"Unsupported mortar_mode '{self.mortar_mode}'. Only 'mesh' and 'SDF' are supported.")

        # add the "ref force arrow in rendering"
        self.cylinder_radius = 0.002
        self.cylinder_length = 0.005
        self.force_cylinder = CylinderObject(
            name="force_cylinder",
            size=(self.cylinder_radius, self.cylinder_length),
            rgba=[1, 0, 0, 1],
            joints=None,
            duplicate_collision_geoms=False,
            obj_type='visual',
        )

        # Load cylinder object
        self.force_cylinder_object = self.force_cylinder.get_obj()
        self.force_cylinder_object.set("pos", "0.1  0.1  0.9")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.mortar)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.mortar,
                x_range=[0, 0.0],
                y_range=[0, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )

        objects = []
        if self.spawn_mortar:
            objects.append(self.mortar)
        objects.append(self.force_cylinder)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

        self.model.merge_assets(self.force_cylinder)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.force_cylinder_body_id = self.sim.model.body_name2id(self.force_cylinder.root_body)
        if self.spawn_mortar:
            self.mortar_body_id = self.sim.model.body_name2id(self.mortar.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=f"{pf}proprio")
        def relative_pose(obs_cache):
            return self._compute_relative_distance()

        @sensor(modality=f"{pf}proprio")
        def relative_wrench(obs_cache):
            return self._compute_relative_wrenches()

        @sensor(modality=f"{pf}proprio")
        def eef_wrench(obs_cache):
            return self.eef_wrench

        @sensor(modality=f"{pf}proprio")
        def eef_pos(obs_cache):
            return self.eef_pos

        @sensor(modality=f"{pf}proprio")
        def eef_rot_ortho6d(obs_cache):
            return T.quat2ortho6(self.eef_quat)

        @sensor(modality=f"{pf}proprio")
        def previous_action(obs_cache):
            return self.previous_action

        sensors = [eef_pos, eef_rot_ortho6d, eef_wrench, relative_pose, relative_wrench, previous_action]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=[True] * len(sensors)
            )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        self.current_waypoint_index = 0
        self.collisions = 0
        self.f_excess = 0
        self.task_space_exits = 0
        self.global_timestep = 0
        self.reward_dict = {
            "force_reward": 0.0,
            "traj_reward": 0.0,
            "action_smoothness": 0.0,
            "step_penalty": 0.0,
            "speed_reward": 0.0,
            "force_total_reward": 0.0,
            "traj_total_reward": 0.0,
            "speed_total_reward": 0.0,
            "action_smoothness_total_reward": 0.0,
        }
        self.translated_action = None

        # Update the contact point visual properties
        self.sim.model._model.vis.scale.contactwidth = 0.01
        self.sim.model._model.vis.scale.contactheight = 0.01

        if self.randomize_reference_trajectory:
            self.reference_trajectory = self._randomize_reference_trajectory(self.control_freq)

        if self.robots[0].composite_controller is None or self.hard_reset:
            # instantiate controllers, only once
            super()._reset_internal()
            if self.ik is None:
                self.ik = MuJoCoIKSolver(self.sim.model, self.sim.data,
                                         "gripper0_right_grip_site",
                                         joint_indexes=self.robots[0].joint_indexes,
                                         position_threshold=0.001,
                                         rotation_threshold=0.01,
                                         max_iterations=1000)

        # Update the initial position of the robot based on the initial pose of the reference trajectory
        if self.reset_with_ik:
            initial_pos = self.reference_trajectory[0][:3]
            initial_pos[2] += 0.001
            result = self.ik.solve_ik(target_pos=initial_pos,
                                      target_rot=T.quat2mat(self.reference_trajectory[0][3:]),
                                      initial_guess=self.init_qpos)

            if result.success:
                self.robots[0].init_qpos = result.joint_angles
            else:
                self.robots[0].init_qpos = self.init_qpos
                print("IK solution not found, using default init_q. Error msg: ", result.message)

        super()._reset_internal()

        self.last_step_time = self.sim.data._data.time

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if self.spawn_mortar and not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.model.body_pos[self.mortar_body_id] = obj_pos
                self.sim.model.body_quat[self.mortar_body_id] = obj_quat

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        # allow episode to finish early if desired
        if self.early_terminations:
            terminated, reason = self._check_terminated()
            info['termination_reason'] = reason
            done = done or terminated

        self.cumulative_reward += reward
        if done:
            self.cumulative_reward = 0.0

        return reward, done, info

    def _update_waypoint_index(self, action):
        # Only update waypoint if we haven't reached the end of trajectory
        if self.current_waypoint_index < self.num_waypoints - 1:

            if self.sim.data._data.time - self.last_step_time > self.step_duration:
                self.global_timestep += 1
                self.previous_action = self.current_action.copy()
                self.current_action = action.copy()
                self.last_step_time = self.sim.data._data.time

                if self.tracking_trajectory_method == 'per_step':  # equivalent to DURATION mode
                    self.current_waypoint_index += 1

                elif self.tracking_trajectory_method == 'per_error_threshold':  # equivalent to TRACKING_ERROR mode
                    if self.tracking_error < self.pose_error_threshold \
                            and self.tracking_force_error < self.force_error_threshold:
                        # Check future waypoints to see if they also satisfy the threshold condition
                        next_waypoint_index = self.current_waypoint_index + 1
                        while next_waypoint_index < self.num_waypoints:
                            # Check if the next waypoint would also satisfy the threshold
                            relative_distance = T.compute_pose_error(self.reference_trajectory[next_waypoint_index], self.eef_pose)
                            next_tracking_error = np.linalg.norm((relative_distance / self.pose_normalization) * self.position_control_dims)

                            # Check force error for next waypoint
                            next_relative_wrench = self.reference_force[next_waypoint_index] - self.eef_wrench
                            next_tracking_force_error = np.linalg.norm((next_relative_wrench / self.force_torque_normalization) * self.force_control_dims)

                            # If the next waypoint doesn't satisfy the threshold, stop
                            if next_tracking_error >= self.pose_error_threshold or \
                               next_tracking_force_error >= self.force_error_threshold:
                                break

                            # Otherwise, continue to the next waypoint
                            next_waypoint_index += 1

                        # Update to the furthest valid waypoint
                        self.current_waypoint_index = min(next_waypoint_index, self.num_waypoints - 1)

                # update in rendering the cylinder representing the normal force/direction
                # assume orientation is given perpendicular to mortar surface
                offset_cylinder_half_size = T.rotate_vector_by_quaternion([0, 0, -self.cylinder_length], self.reference_trajectory[self.current_waypoint_index][3:])
                self.sim.model.body_pos[self.force_cylinder_body_id] = self.reference_trajectory[self.current_waypoint_index][:3] + offset_cylinder_half_size
                self.sim.model.body_quat[self.force_cylinder_body_id] = T.convert_quat(self.reference_trajectory[self.current_waypoint_index][3:], "wxyz")

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this environment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        # Single robot
        robot = self.robots[0]

        # Verify that the action is the correct dimension
        assert len(action) == robot.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            robot.action_dim, len(action)
        )

        robot.control(action, policy_step=policy_step)

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Task space limit reached
            - Task completion (tracking completed)

        Returns:
            bool: True if episode is terminated
        """
        terminated = False
        reason = ""

        if (self.timestep >= self.horizon) and not self.ignore_done:
            terminated = True
            reason = "HORIZON REACHED"

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            terminated = True
            reason = "COLLIDED"

        # Prematurely terminate if the end effector leave the play area
        if not self._check_task_space_limits():
            terminated = True
            reason = "TASK SPACE LIMIT REACHED"

        # Prematurely terminate if task is completed
        if self._check_success():
            terminated = True
            reason = "TRACKING COMPLETED"

        if self._check_waypoint_completion_delay():
            terminated = True
            reason = "WAYPOINT COMPLETION DELAY REACHED"

        return terminated, reason

    def _check_success(self):
        """
            Check if trajectory tracking is completed

        Returns:
            bool: True completed task
        """

        return self.current_waypoint_index + 1 == self.num_waypoints

    def _check_task_space_limits(self):
        """
        Check if the eef is not too far away from mortar, works because mortar space center is at [0,0,0], does it need generalization?

        Returns:
            bool: True within task box space limits
        """

        ee_pos = self.robots[0].recent_ee_pose['right'].current[:3]
        return not np.any(np.abs(ee_pos) > self.task_box)

    def _check_force_torque_limits(self):
        """
        Check that the robot is not exerting too much force/torque

        Returns:
            bool: True within force/torque limits
        """
        abs_ft = np.abs(self.eef_wrench)
        return not np.any(abs_ft > self.force_torque_limits)

    def _check_waypoint_completion_delay(self):
        """
        Check if the waypoint completion delay has been reached
        """
        # Convert waypoint_completion_delay from seconds to timesteps
        # Since num_waypoints = control_freq * duration // 10, each waypoint represents (10/control_freq) seconds
        if self.waypoint_completion_delay == 0:
            return False
        delay_in_timesteps = self.waypoint_completion_delay / self.seconds_per_waypoint
        delay = self.global_timestep - self.current_waypoint_index
        # print(f"delay: {delay}, delay_in_timesteps: {delay_in_timesteps}")
        return delay > delay_in_timesteps

    def _randomize_reference_trajectory(self, control_freq):
        mortar_diameter = self.task_config["mortar_diameter"]
        mortar_inner_height = self.task_config["mortar_inner_height"]
        max_inclination_angle = self.task_config["max_inclination_angle"]
        initial_orientation = self.task_config["initial_orientation"]
        initial_position = self.task_config["initial_position"].copy()
        initial_position[2] += mortar_inner_height  # Add inner height to z position

        if self.randomize_reference_trajectory:
            # randomize the duration and the number of waypoints
            self.duration = int(np.random.uniform(low=self.duration_range[0], high=self.duration_range[1]))
            self.num_waypoints = int(control_freq * self.duration // 10)
            # randomize the desired height
            desired_height = np.random.uniform(low=0.001, high=0.020)
            # update the target force
            self.target_force = int(np.random.uniform(low=self.target_force_range[0], high=self.target_force_range[1]))
            self.reference_force = np.array([[0, 0, self.target_force, 0, 0, 0]] * self.num_waypoints)

        else:
            desired_height = self.task_config["desired_height"]
            self.target_force = self.task_config["target_force"]
        # print(f"duration: {self.duration}, num_waypoints: {self.num_waypoints}, target_force: {target_force} desired_height: {desired_height}")

        reference_trajectory = generate_mortar_trajectory(
            mortar_diameter=mortar_diameter,
            desired_height=desired_height,
            n_steps=self.num_waypoints,
            default_quat=initial_orientation,
            max_angle=max_inclination_angle
        )
        reference_trajectory[:, :3] += initial_position

        # self.max_step_size = compute_max_step_size(reference_trajectory) * 5
        self.max_step_size = self.pose_normalization
        self.pose_error_threshold = np.linalg.norm(self.tracking_trajectory_threshold * self.position_control_dims / self.max_step_size)
        self.force_error_threshold = np.linalg.norm(self.tracking_force_threshold * self.force_control_dims / self.force_torque_normalization)
        return reference_trajectory

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self.action_ndim

    @property
    def current_wrench(self):
        return self.robots[0].composite_controller.part_controllers['right'].current_wrench

    @property
    def eef_wrench(self):
        return self.robots[0].composite_controller.part_controllers['right'].eef_wrench

    @property
    def eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id['right']])

    @property
    def eef_quat(self):
        return T.mat2quat(self.sim.data.site_xmat[self.robots[0].eef_site_id['right']].reshape(3, 3))

    @property
    def eef_pose(self):
        return np.concatenate([self.eef_pos, self.eef_quat])

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        low, high = -np.ones(self.action_ndim), np.ones(self.action_ndim)
        return low, high
