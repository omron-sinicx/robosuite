from copy import deepcopy, copy
import multiprocessing
import numpy as np
from robosuite.models.arenas.osx_wipe_arena import OSXWipeArena
from robosuite.models.objects.composite.hammer import HammerObject
from robosuite.utils.mjcf_utils import xml_path_completion

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

# Default Wipe environment configuration
DEFAULT_WIPE_CONFIG = {
    # settings for reward
    "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
    "wipe_contact_reward": 0.01,  # reward for contacting something with the wiping tool
    "unit_wiped_reward": 50.0,  # reward per peg wiped
    "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.05,  # penalty for each step that the force is over the safety threshold
    "distance_multiplier": 5.0,  # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
    "distance_th_multiplier": 5.0,  # multiplier in the tanh function for the aforementioned reward
    # settings for table top
    "table_friction": [0.03, 0.005, 0.0001],  # Friction parameters for the table
    "table_height": 0.0,  # Additional height of the table over the default location
    "table_height_std": 0.0,  # Standard deviation to sample different heigths of the table each episode
    "line_width": 0.04,  # Width of the line to wipe (diameter of the pegs)
    "two_clusters": False,  # if the dirt to wipe is one continuous line or two
    "coverage_factor": 0.6,  # how much of the table surface we cover
    "num_markers": 100,  # How many particles of dirt to generate in the environment
    # settings for thresholds
    "contact_threshold": 1.0,  # Minimum eef force to qualify as contact [N]
    "pressure_threshold": 0.5,  # force threshold (N) to overcome to get increased contact wiping reward
    "pressure_threshold_max": 60.0,  # maximum force allowed (N)
    # misc settings
    "print_results": False,  # Whether to print results or not
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,  # Whether we allow for early terminations or not
    "use_condensed_obj_obs": True,  # Whether to use condensed object observation representation (only applicable if obj obs is active)
    "spawn_hammer": True,  # whether to spawn a hammer on top of the wiping marks
}


class TwoArmWiping(TwoArmEnv):
    """
    This class corresponds to the lifting task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" corresponds to either "bimanual" if a bimanual robot is used or "single-arm-opposed" if two
        single-arm robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

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

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

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

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        controller_configs=None,
        initialization_noise="default",
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        task_config=DEFAULT_WIPE_CONFIG,
        ** kwargs,
    ):

        self.controller_configs = controller_configs

        # Get config
        self.task_config = task_config

        # settings for table top
        self.table_friction = table_friction

        # Set task-specific parameters
        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config["arm_limit_collision_penalty"]
        self.wipe_contact_reward = self.task_config["wipe_contact_reward"]
        self.unit_wiped_reward = self.task_config["unit_wiped_reward"]
        self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        self.excess_force_penalty_mul = self.task_config["excess_force_penalty_mul"]
        self.distance_multiplier = self.task_config["distance_multiplier"]
        self.distance_th_multiplier = self.task_config["distance_th_multiplier"]
        # Final reward computation
        # So that is better to finish that to stay touching the table for 100 steps
        # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
        self.task_complete_reward = self.unit_wiped_reward * (self.wipe_contact_reward + 0.5)
        # Verify that the distance multiplier is not greater than the task complete reward
        assert (
            self.task_complete_reward > self.distance_multiplier
        ), "Distance multiplier cannot be greater than task complete reward!"

        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold = self.task_config["pressure_threshold"]
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]

        # settings for tabletop
        self.line_width = self.task_config["line_width"]
        self.two_clusters = self.task_config["two_clusters"]
        self.coverage_factor = self.task_config["coverage_factor"]
        self.num_markers = self.task_config["num_markers"]

        # misc settings
        self.print_results = self.task_config["print_results"]
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.use_contact_obs = self.task_config["use_contact_obs"]
        self.early_terminations = self.task_config["early_terminations"]
        self.use_condensed_obj_obs = self.task_config["use_condensed_obj_obs"]
        self.spawn_hammer = self.task_config["spawn_hammer"]

        # Scale reward if desired (see reward method for details)
        self.reward_normalization_factor = horizon / (
            self.num_markers * self.unit_wiped_reward + horizon * (self.wipe_contact_reward + self.task_complete_reward)
        )

        # set other wipe-specific attributes
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0
        self.metadata = []
        self.spec = "spec"

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # set after init to ensure self.robots is set
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        super().__init__(
            robots=['UR5e', 'UR5e'],
            env_configuration='single-arm-opposed',
            controller_configs=controller_configs,
            base_types="NullMount",
            gripper_types=["Robotiq140Gripper", "WipingGripper"],
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
        )

    def step(self, action_dict):
        """
            Format action to what the environment expects:
            expected action: [robot0 stiffness, robot0 position, robot0 rotation axis angle/delta, robot0 gripper,
                             robot1 stiffness, robot1 position, robot1 rotation axis angle/delta]

            arg: `action_dict`: dict or list.
                If `dict`, expect lerobot format.
                If `list`, expect environment format
        """
        if isinstance(action_dict, dict):
            action_d = copy(action_dict)  # do not modify original dict
            if self.controller_configs['body_parts']['right']['type'] == 'JOINT_POSITION':
                action_d = split_actions(action_d)

                action = np.concatenate([
                    action_d['action.qpos'][0],
                    action_d['action.gripper'],
                    action_d['action.qpos'][1],
                ])
            elif self.controller_configs['body_parts']['right']['type'] == 'OSC_POSE':
                # Convert rotation to axis angle if necessary
                if 'action.rotation_ortho6' in action_d:
                    action_d['action.rotation_axis_angle'] = [T.ortho62axisangle(action_d['action.rotation_ortho6'][0]),
                                                              T.ortho62axisangle(action_d['action.rotation_ortho6'][1])]

                # action_d = split_actions(action_d)

                if self.controller_configs['body_parts']['right']['impedance_mode'] == 'fixed':
                    action = np.concatenate([
                        action_d['action.position'][0],
                        action_d['action.rotation_axis_angle'][0],
                        action_d['action.gripper'][0],
                        action_d['action.position'][1],
                        action_d['action.rotation_axis_angle'][1],
                    ])
                else:
                    # Get the expected stiffness format depending on the controller's impedance mode
                    stiffness_type = 'cholesky' if self.controller_configs['body_parts']['right']['impedance_mode'] == 'variable_full_kp' else 'diag'
                    stiffness_key = f'action.stiffness_{stiffness_type}'

                    action = np.concatenate([
                        action_d[stiffness_key][0],
                        action_d['action.position'][0],
                        action_d['action.rotation_axis_angle'][0],
                        action_d['action.gripper'][0],
                        action_d[stiffness_key][1],
                        action_d['action.position'][1],
                        action_d['action.rotation_axis_angle'][1],
                    ])
            else:
                raise ValueError(f"Unsupported controller type: {self.controller_configs['type']}. Only 'JOINT_POSITION' and 'OSC_POSE' are supported.")
        else:
            action = action_dict

        return super().step(action)

    def _get_active_markers(self, c_geoms):
        """
        Get the markers that are currently being wiped by the tool

        Args:
            c_geoms (list): List of corner geoms for the tool

        Returns:
            list: List of active markers
        """
        active_markers = []
        corner1_id = self.sim.model.geom_name2id(c_geoms[0])
        corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
        corner2_id = self.sim.model.geom_name2id(c_geoms[1])
        corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
        corner3_id = self.sim.model.geom_name2id(c_geoms[2])
        corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
        corner4_id = self.sim.model.geom_name2id(c_geoms[3])
        corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])

        # Unit vectors on my plane
        v1 = corner1_pos - corner2_pos
        v1 /= np.linalg.norm(v1)
        v2 = corner4_pos - corner2_pos
        v2 /= np.linalg.norm(v2)

        # Corners of the tool in the coordinate frame of the plane
        t1 = np.array([np.dot(corner1_pos - corner2_pos, v1), np.dot(corner1_pos - corner2_pos, v2)])
        t2 = np.array([np.dot(corner2_pos - corner2_pos, v1), np.dot(corner2_pos - corner2_pos, v2)])
        t3 = np.array([np.dot(corner3_pos - corner2_pos, v1), np.dot(corner3_pos - corner2_pos, v2)])
        t4 = np.array([np.dot(corner4_pos - corner2_pos, v1), np.dot(corner4_pos - corner2_pos, v2)])

        pp = [t1, t2, t4, t3]

        # Normal of the plane defined by v1 and v2
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)

        def isLeft(P0, P1, P2):
            return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

        def PointInRectangle(X, Y, Z, W, P):
            return isLeft(X, Y, P) < 0 and isLeft(Y, Z, P) < 0 and isLeft(Z, W, P) < 0 and isLeft(W, X, P) < 0

        # Only go into this computation if there are contact points
        print("ncon", self.sim.data.ncon)
        if self.sim.data.ncon != 0:

            # Check each marker that is still active
            for marker in self.model.mujoco_arena.markers:

                # Current marker 3D location in world frame
                marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

                # We use the second tool corner as point on the plane and define the vector connecting
                # the marker position to that point
                v = marker_pos - corner2_pos

                # Shortest distance between the center of the marker and the plane
                dist = np.dot(v, n)

                # Projection of the center of the marker onto the plane
                projected_point = np.array(marker_pos) - dist * n

                # Positive distances means the center of the marker is over the plane
                # The plane is aligned with the bottom of the wiper and pointing up, so the marker would be over it
                if dist > 0.0:
                    # Distance smaller than this threshold means we are close to the plane on the upper part
                    if dist < 0.02:
                        # Write touching points and projected point in coordinates of the plane
                        pp_2 = np.array(
                            [np.dot(projected_point - corner2_pos, v1), np.dot(projected_point - corner2_pos, v2)]
                        )
                        # Check if marker is within the tool center:
                        if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
                            active_markers.append(marker)
        return active_markers

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of self.unit_wiped_reward is provided per single dirt (peg) wiped during this step
            - a discrete reward of self.task_complete_reward is provided if all dirt is wiped

        Note that if the arm is either colliding or near its joint limit, a reward of 0 will be automatically given

        Un-normalized summed components if using reward shaping (individual components can be set to 0:

            - Reaching: in [0, self.distance_multiplier], proportional to distance between wiper and centroid of dirt
              and zero if the table has been fully wiped clean of all the dirt
            - Table Contact: in {0, self.wipe_contact_reward}, non-zero if wiper is in contact with table
            - Wiping: in {0, self.unit_wiped_reward}, non-zero for each dirt (peg) wiped during this step
            - Cleaned: in {0, self.task_complete_reward}, non-zero if no dirt remains on the table
            - Collision / Joint Limit Penalty: in {self.arm_limit_collision_penalty, 0}, nonzero if robot arm
              is colliding with an object
              - Note that if this value is nonzero, no other reward components can be added
            - Large Force Penalty: in [-inf, 0], scaled by wiper force and directly proportional to
              self.excess_force_penalty_mul if the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0], scaled by estimated wiper acceleration and directly
              proportional to self.ee_accel_penalty

        Note that the final per-step reward is normalized given the theoretical best episode return and then scaled:
        reward_scale * (horizon /
        (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        total_force_ee = max(
            [
                np.linalg.norm(np.array(self.robots[1].recent_ee_forcetorques[arm].current[:3]))
                for arm in self.robots[1].arms
            ]
        )

        # Neg Reward from collisions of the arm with the table
        if self.check_contact(self.robots[1].robot_model):
            print("in contact")
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self.robots[1].check_q_limits():
            print("in joint limits")
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        else:
            print("not colliding or in joint limits")
            # If the arm is not colliding or in joint limits, we check if we are wiping
            # (we don't want to reward wiping if there are unsafe situations)
            active_markers = []

            # Current 3D location of the corners of the wiping tool in world frame
            for arm in self.robots[1].arms:
                c_geoms = self.robots[1].gripper[arm].important_geoms["corners"]
                active_markers += self._get_active_markers(c_geoms)

            # Obtain the list of currently active (wiped) markers that where not wiped before
            # These are the markers we are wiping at this step
            lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
            new_active_markers = np.array(active_markers)[lall]

            # Loop through all new markers we are wiping at this step
            for new_active_marker in new_active_markers:
                # Grab relevant marker id info
                new_active_marker_geom_id = self.sim.model.geom_name2id(new_active_marker.visual_geoms[0])
                # Make this marker transparent since we wiped it (alpha = 0)
                self.sim.model.geom_rgba[new_active_marker_geom_id][3] = 0
                # Add this marker the wiped list
                self.wiped_markers.append(new_active_marker)
                # Add reward if we're using the dense reward
                if self.reward_shaping:
                    reward += self.unit_wiped_reward

            # Additional reward components if using dense rewards
            if self.reward_shaping:
                # If we haven't wiped all the markers yet, add a smooth reward for getting closer
                # to the centroid of the dirt to wipe
                if len(self.wiped_markers) < self.num_markers:
                    _, _, mean_pos_to_things_to_wipe = self._get_wipe_information()
                    mean_distance_to_things_to_wipe = np.linalg.norm(mean_pos_to_things_to_wipe)
                    reward += self.distance_multiplier * (
                        1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe)
                    )

                # Reward for keeping contact
                if self.sim.data.ncon != 0 and self._has_gripper_contact:
                    reward += self.wipe_contact_reward

                # Penalty for excessive force with the end-effector
                if total_force_ee > self.pressure_threshold_max:
                    reward -= self.excess_force_penalty_mul * total_force_ee
                    self.f_excess += 1

                # Reward for pressing into table
                # TODO: Need to include this computation somehow in the scaled reward computation
                elif total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1:
                    reward += self.wipe_contact_reward + 0.01 * total_force_ee
                    if self.sim.data.ncon > 50:
                        reward += 10.0 * self.wipe_contact_reward

                # Penalize large accelerations
                reward -= self.ee_accel_penalty * max(
                    [np.mean(abs(self.robots[1].recent_ee_acc[arm].current)) for arm in self.robots[1].arms]
                )

            # Final reward if all wiped
            if len(self.wiped_markers) == self.num_markers:
                reward += self.task_complete_reward

        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}"
                "wiped markers: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    ws=len(self.wiped_markers),
                    sc=self.collisions,
                    fe=self.f_excess,
                )
            )
            print(string_to_print)

        # If we're scaling our reward, we normalize the per-step rewards given the theoretical best episode return
        # This is equivalent to scaling the reward by:
        #   reward_scale * (horizon /
        #       (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))
        if self.reward_scale:
            reward *= self.reward_scale * self.reward_normalization_factor
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Set up robots facing towards each other by rotating them from their default position
        # a_bot
        self.robots[0].robot_model.set_base_xpos((-0.003, -0.527, 0.750))
        self.robots[0].robot_model.set_base_ori((0, 0, 0))
        self.robots[0].init_qpos = np.array([1.390, -1.949, 2.020, -1.643, -1.573, 0.0])
        # b_bot
        self.robots[1].robot_model.set_base_xpos((0.003,  0.525, 0.750))
        self.robots[1].robot_model.set_base_ori((0, 0, -np.pi))
        # self.robots[1].init_qpos = np.array([0.831, -1.666, 2.364, -2.291, -1.585, -2.313])
        self.robots[1].init_qpos = np.array([0.831, -1.666, 2.364, -2.291, -1.585, -0.643])  # fix initial pose for camera
        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[1].robot_model.contact_geoms

        # load model for table top workspace
        mujoco_arena = OSXWipeArena(
            table_friction=self.table_friction,
            wiping_area=(0.15, 0.15, 0.05),
            center_pose=[-0.175, 0.0],
            num_markers=30,
            line_width=0.05,
            coverage_factor=0.8,
            seed=0,  # Random seed
            xml=xml_path_completion("arenas/osx_arena.xml")
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        if self.spawn_hammer:
            # initialize objects of interest
            self.hammer = HammerObject(
                name="hammer",
                handle_radius=(0.01, 0.015),
                handle_length=(0.1, 0.125))

            # Create placement initializer
            if self.placement_initializer is not None:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(self.hammer)
            else:
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=self.hammer,
                    x_range=[0, 0],
                    y_range=[0, 0],
                    rotation=np.deg2rad([90, 105]),
                    rotation_axis="y",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=np.array((-0.15, 0.0, 0.87)),
                )
        else:
            self.hammer = None

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hammer,
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robots[1].robot_model.naming_prefix
        modality = "object"

        sensors = []
        names = []

        # Add binary contact observation
        if self.use_contact_obs:

            @sensor(modality=f"{pf}proprio")
            def gripper_contact(obs_cache):
                return self._has_gripper_contact

            sensors.append(gripper_contact)
            names.append(f"{pf}contact")

        # object information in the observation
        if self.use_object_obs:

            if self.use_condensed_obj_obs:
                # use implicit representation of wiping objects
                @sensor(modality=modality)
                def wipe_radius(obs_cache):
                    wipe_rad, wipe_cent, _ = self._get_wipe_information()
                    obs_cache["wipe_centroid"] = wipe_cent
                    return wipe_rad

                @sensor(modality=modality)
                def wipe_centroid(obs_cache):
                    return obs_cache["wipe_centroid"] if "wipe_centroid" in obs_cache else np.zeros(3)

                @sensor(modality=modality)
                def proportion_wiped(obs_cache):
                    return len(self.wiped_markers) / self.num_markers

                sensors += [proportion_wiped, wipe_radius, wipe_centroid]
                names += ["proportion_wiped", "wipe_radius", "wipe_centroid"]

                if self.use_robot_obs:
                    # also use ego-centric obs
                    @sensor(modality=modality)
                    def gripper_to_wipe_centroid(obs_cache):
                        return (
                            obs_cache["wipe_centroid"] - obs_cache[f"{pf}eef_pos"]
                            if "wipe_centroid" in obs_cache and f"{pf}eef_pos" in obs_cache
                            else np.zeros(3)
                        )

                    sensors.append(gripper_to_wipe_centroid)
                    names.append("gripper_to_wipe_centroid")

            else:
                # use explicit representation of wiping objects
                for i, marker in enumerate(self.model.mujoco_arena.markers):
                    marker_sensors, marker_sensor_names = self._create_marker_sensors(i, marker, modality)
                    sensors += marker_sensors
                    names += marker_sensor_names

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_marker_sensors(self, i, marker, modality="object"):
        """
        Helper function to create sensors for a given marker. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            i (int): ID number corresponding to the marker
            marker (MujocoObject): Marker to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given marker
                names (list): array of corresponding observable names
        """
        pf = self.robots[1].robot_model.naming_prefix

        @sensor(modality=modality)
        def marker_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

        @sensor(modality=modality)
        def marker_wiped(obs_cache):
            return [0, 1][marker in self.wiped_markers]

        sensors = [marker_pos, marker_wiped]
        names = [f"marker{i}_pos", f"marker{i}_wiped"]

        if self.use_robot_obs:
            # also use ego-centric obs
            @sensor(modality=modality)
            def gripper_to_marker(obs_cache):
                return (
                    obs_cache[f"marker{i}_pos"] - obs_cache[f"{pf}eef_pos"]
                    if f"marker{i}_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors.append(gripper_to_marker)
            names.append(f"gripper_to_marker{i}")

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena(self.sim)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
            if self.spawn_hammer:

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Reset all internal vars for this wipe task
        self.timestep = 0
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0

        # set after init to ensure self.robots is set
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # Color the gripper visualization site according to its distance to each handle
        # if vis_settings["grippers"]:
        #     handles = [self.pot.important_sites[f"handle{i}"] for i in range(2)]
        #     grippers = (
        #         [self.robots[0].gripper[arm] for arm in self.robots[0].arms]
        #         if self.env_configuration == "bimanual"
        #         else [robot.gripper for robot in self.robots]
        #     )
        #     for gripper, handle in zip(grippers, handles):
        #         self._visualize_gripper_to_target(gripper=gripper, target=handle, target_type="site")

    def _check_success(self):
        """
        Checks if Task succeeds (all dirt wiped).

        Returns:
            bool: True if completed task
        """
        return True if len(self.wiped_markers) == self.num_markers else False

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion (wiping succeeded)
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[1].robot_model):
            if self.print_results:
                print(40 * "-" + " COLLIDED " + 40 * "-")
            terminated = True

        # Prematurely terminate if task is success
        if self._check_success():
            if self.print_results:
                print(40 * "+" + " FINISHED WIPING " + 40 * "+")
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self.robots[1].check_q_limits():
            if self.print_results:
                print(40 * "-" + " JOINT LIMIT " + 40 * "-")
            terminated = True

        return terminated

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

        # Update force bias
        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[1].ee_force['right']
            self.ee_torque_bias = self.robots[1].ee_torque['right']

        if self.get_info:
            info["add_vals"] = ["nwipedmarkers", "colls", "percent_viapoints_", "f_excess"]
            info["nwipedmarkers"] = len(self.wiped_markers)
            info["colls"] = self.collisions
            info["percent_viapoints_"] = len(self.wiped_markers) / self.num_markers
            info["f_excess"] = self.f_excess

        # allow episode to finish early if allowed
        if self.early_terminations:
            done = done or self._check_terminated()

        return reward, done, info

    def _get_wipe_information(self):
        """Returns set of wiping information"""
        mean_pos_to_things_to_wipe = np.zeros(3)
        wipe_centroid = np.zeros(3)
        marker_positions = []
        num_non_wiped_markers = 0
        if len(self.wiped_markers) < self.num_markers:
            for marker in self.model.mujoco_arena.markers:
                if marker not in self.wiped_markers:
                    marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])
                    wipe_centroid += marker_pos
                    marker_positions.append(marker_pos)
                    num_non_wiped_markers += 1
            wipe_centroid /= max(1, num_non_wiped_markers)

            # Mean position to things to wipe to the closest arm
            mean_pos_to_things_to_wipe_list = [wipe_centroid - self._get_eef_xpos(arm) for arm in self.robots[1].arms]
            mean_pos_to_things_to_wipe = mean_pos_to_things_to_wipe_list[
                np.argmin([np.linalg.norm(x) for x in mean_pos_to_things_to_wipe_list])
            ]
        # Radius of circle from centroid capturing all remaining wiping markers
        max_radius = 0
        if num_non_wiped_markers > 0:
            max_radius = np.max(np.linalg.norm(np.array(marker_positions) - wipe_centroid, axis=1))
        # Return all values
        return max_radius, wipe_centroid, mean_pos_to_things_to_wipe

    def _get_eef_xpos(self, arm):
        """
        Grabs End Effector position as specifed by the arm argument

        Args:
            arm (str): Arm name

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id[arm]])

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the any of the grippers are making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        return any(
            [
                np.linalg.norm(self.robots[1].ee_force[arm] - self.ee_force_bias) > self.contact_threshold
                for arm in self.robots[1].arms
            ]
        )


def split_actions(action_dict: dict):
    """
        Expect data for a bimanual set up so let's split the data
    """
    res = copy(action_dict)
    for key, value in action_dict.items():
        if len(value) % 2 == 0:
            res[key] = np.split(value, 2)
        else:
            res[key] = value
    return res
