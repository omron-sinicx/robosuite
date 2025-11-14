from copy import copy
import numpy as np

from robosuite.utils.binding_utils import MjSim

from robosuite.utils.ik_solver import MuJoCoIKSolver
import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.controller import Controller
from robosuite.controllers.parts.generic.joint_pos import JointPositionController
from robosuite.controllers.parts.generic.joint_vel import JointVelocityController
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import *

try:
    from ur_pykdl.ik_solver import IKSolver
except ImportError:
    IKSolver = None


# Supported impedance modes
COMPLIANCE_MODES = {"fixed", "variable_stiffness", "variable_stiffness_and_p_gains"}


class ForwardDynamicsComplianceController(Controller):
    """
    Controller for hybrid force-position control of robot arms. Combines position/orientation control with force/torque control
    in operational space.

    The controller operates in either end-effector ('eef') or robot base ('robot_base') frame and uses:
    - Position/orientation control through stiffness-based pose error
    - Force/torque control through wrench error
    - PD control for dynamic response

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        ref_name (str): Name of controlled robot arm's reference frame (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:
            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        inner_controller_config (dict): Configuration for the inner controller, containing:
            :`'type'`: Type of inner controller - one of "JOINT_POSITION", "JOINT_VELOCITY", or "OSC_POSE"
            :`'input_max'`, `'input_min'`, `'output_max'`, `'output_min'`: Control input/output limits
            :Additional controller-specific parameters

        iterations (int): Number of iterations to run the controller for each step (default: 1)

        error_scale (float): Scaling factor applied to the computed error (default: 1.0)

        stiffness (float or array): Cartesian stiffness values for each dimension (default: 500)

        kp (float or array): Proportional gains for PD control (default: 0.1)

        kd (float or array): Derivative gains for PD control (default: 0.0)

        compliance_mode (str): Mode of compliance control. One of:
            :`'fixed'`: Fixed stiffness values
            :`'variable_stiffness'`: Variable diagonal stiffness matrix
            :`'variable_stiffness_and_p_gains'`: Variable stiffness and P gains

        policy_freq (int): Control policy frequency in Hz (default: 20)

        force_limits (2-tuple): Min/max force limits in N (default: (-50.0, 50.0))

        torque_limits (2-tuple): Min/max torque limits in Nm (default: (-10.0, 10.0))

        ft_buffer_size (int): Size of force/torque measurement buffer (default: 10)

        stiffness_limits (2-tuple): Min/max stiffness values (default: (50, 500))

        kp_limits (2-tuple): Min/max proportional gain values (default: (0, 300))

        damping_ratio_limits (2-tuple): Min/max damping ratio values (default: (0, 100))

        selection_matrix (array): 6D vector specifying which dimensions to control with position vs force (default: ones(6))

        position_limits (array or None): Position limits for the end-effector (default: None)

        orientation_limits (array or None): Orientation limits for the end-effector (default: None)

        interpolator_pos (Interpolator): Position interpolator for smooth transitions (default: None)

        interpolator_ori (Interpolator): Orientation interpolator for smooth transitions (default: None)

        control_delta (bool): If True, interpret control inputs as relative changes (default: True)

        gripper_body_name (str): Name of gripper body for payload compensation (default: None)

        frame_of_reference (str): Frame for control computations - either "eef" or "robot_base" (default: "eef")

        lite_physics (bool): Whether to use simplified physics computations (default: True)

        use_kdl (bool): Whether to use KDL for inverse kinematics (default: False)

    Raises:
        AssertionError: If an invalid compliance_mode is specified
        ValueError: If an invalid inner_controller_type is specified
    """

    def __init__(
        self,
        sim: MjSim,
        ref_name,
        joint_indexes,
        actuator_range,
        inner_controller_config,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        iterations=1,
        error_scale=1.0,
        stiffness=500,
        kp=0.1,
        kd=0.0,
        compliance_mode="fixed",
        policy_freq=20,
        force_limits=(-50.0, 50.0),
        torque_limits=(-10.0, 10.0),
        ft_buffer_size=10,
        stiffness_limits=(50, 500),
        kp_limits=(0, 300),
        damping_ratio=1.0,
        damping_ratio_limits=(0, 100),
        virtual_force_limits=(-50.0, 50.0),
        selection_matrix=np.ones(6),
        position_limits=None,
        orientation_limits=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_delta=True,
        gripper_body_name=None,  # If none, do not compensate payload
        frame_of_reference="eef",  # or "robot_base"
        lite_physics=True,
        use_kdl=False,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        self.use_kdl = use_kdl
        self.iterations = iterations
        self.ft_prefix = ref_name.split('_')[0] + '_' + kwargs.get("part_name", None)
        self.frame_of_reference = frame_of_reference
        self.selection_matrix = selection_matrix
        self.virtual_force = np.zeros(6)

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            lite_physics=lite_physics,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
            ft_buffer_size=ft_buffer_size,
            gripper_body_name=gripper_body_name,
        )

        # Instantiate the inner position/velocity controller
        self.inner_controller_type = inner_controller_config["type"]
        if self.inner_controller_type == "JOINT_POSITION":
            inner_controller_class = JointPositionController
        elif self.inner_controller_type == "JOINT_VELOCITY":
            inner_controller_class = JointVelocityController
        elif self.inner_controller_type == "OSC_POSE":
            inner_controller_class = OperationalSpaceController
        else:
            raise ValueError("Invalid inner_controller_config type")

        inner_controller_config['control_delta'] = True

        self.inner_controller = inner_controller_class(
            sim=sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            part_name=self.part_name,
            naming_prefix=self.naming_prefix,
            **inner_controller_config,
        )

        # Verify the proposed impedance mode is supported
        assert compliance_mode in COMPLIANCE_MODES, (
            "Error: Tried to instantiate Compliance controller for unsupported "
            "compliance mode! Inputted compliance mode: {}, Supported modes: {}".format(compliance_mode, COMPLIANCE_MODES)
        )
        self.compliance_mode = compliance_mode

        self.use_delta = control_delta
        self.control_pose_dim = 6 if control_delta else 7  # desired position/orientation
        self.control_dim = self.control_pose_dim
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        self.control_dim += 6  # + force/torque
        self.force_min = self.nums2array(force_limits[0], 3)
        self.force_max = self.nums2array(force_limits[1], 3)
        self.torque_min = self.nums2array(torque_limits[0], 3)
        self.torque_max = self.nums2array(torque_limits[1], 3)

        self.stiffness = self.nums2array(stiffness, 6)
        self.stiffness_limits = np.array(stiffness_limits)
        # stiffness limits
        self.stiffness_min = self.nums2array(stiffness_limits[0], 6)
        self.stiffness_max = self.nums2array(stiffness_limits[1], 6)

        # Add to control dim based on compliance_mode
        if self.compliance_mode == "variable_stiffness":
            self.control_dim += 6
        elif self.compliance_mode == "variable_stiffness_and_p_gains":
            self.control_dim += 12

        self.kp = self.nums2array(kp, 6)
        self.kd = self.nums2array(kd, 6)
        # kp and kd limits
        self.kp_limits = np.array(kp_limits)
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio = damping_ratio
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # virtual force limits
        self.virtual_force_min = self.nums2array(virtual_force_limits[0], 6)
        self.virtual_force_max = self.nums2array(virtual_force_limits[1], 6)

        self.error_scale = error_scale

        self.last_err = np.zeros(6)

        # limits
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq
        self.period = self.model_timestep

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # initialize
        self.goal_pose = None  # Goal velocity desired, pre-compensation
        self.desired_force_torque = np.zeros(6)

        if self.use_kdl:
            self.kdl_solver = IKSolver(robot='ur5e_powder_grinding_default', rospackage='osx_powder_grinding',
                                       base_link='base_link', ee_link='gripper_tip_link')
            self.kdl_solver.build_generic_model()
            self.mjc_ik_solver = MuJoCoIKSolver(
                                                self.sim.model.get_xml(),
                                                [],
                                                ref_name,
                                                position_threshold=0.001,
                                                rotation_threshold=0.01,
                                                time_limit=0.1,
                                                joint_indexes=self.qpos_index,
                                                base_body_name=None
                                            )

    def update(self):
        super().update()

        # Synchronize Joint Positions
        self.current_joint_positions = self.joint_pos
        self.last_joint_positions = copy(self.joint_pos)
        self.current_joint_velocities = self.joint_vel
        self.last_joint_velocities = copy(self.joint_vel)

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets the controller's goal state based on the input action. Processes the action according to the compliance mode
        and updates internal controller parameters.

        Args:
            action (np.array): Control action array with format depending on compliance_mode:
                - 'fixed': [delta_pose (6), desired_wrench (6)]
                - 'variable_stiffness': [delta_pose (6), desired_wrench (6), stiffness (6)]
                - 'variable_stiffness_and_p_gains': [delta_pose (6), desired_wrench (6), stiffness (6), kp (6)]

            set_pos (np.array, optional): If provided, directly sets the absolute goal position, overriding action
            set_ori (np.array, optional): If provided, directly sets the absolute goal orientation as a rotation matrix

        Note:
            - For delta_pose: First 3 values are position, last 3 are axis-angle orientation
            - For desired_wrench: First 3 values are force (N), last 3 are torque (Nm)
            - Stiffness and kp values are clipped to their configured limits
        """
        # Update state
        self.update()

        if self.compliance_mode == "variable_stiffness":
            delta, desired_ft, stiffness = action[:self.control_pose_dim], action[self.control_pose_dim:12], action[12:]
            self.stiffness = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
        elif self.compliance_mode == "variable_stiffness_and_p_gains":
            delta, desired_ft, stiffness, kp = action[:self.control_pose_dim], action[self.control_pose_dim:12], action[12:18], action[18:]
            self.stiffness = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
        else:  # This is case "fixed"
            delta, desired_ft = action[:self.control_pose_dim], action[self.control_pose_dim:]

        desired_ft[:3] = np.clip(desired_ft[:3], self.force_min, self.force_max)
        desired_ft[3:] = np.clip(desired_ft[3:], self.torque_min, self.torque_max)
        self.desired_force_torque = desired_ft

        # If we're using deltas, interpret actions as such
        if self.use_delta:
            scaled_delta = self.scale_action(delta)
        # Else, interpret actions as absolute values
        else:
            if set_pos is None:
                set_pos = delta[:3]
            if set_ori is None:
                set_ori = (T.quat2mat(delta[3:7]))
            # No scaling of values since these are absolute values
            scaled_delta = np.zeros_like(delta)

        # We only want to update goal orientation if there is a valid delta ori value OR if we're using absolute ori
        self.goal_ori = set_goal_orientation(
            scaled_delta[3:], self.ref_ori_mat, orientation_limit=self.orientation_limits, set_ori=set_ori
        )
        self.goal_pos = set_goal_position(
            scaled_delta[:3], self.ref_pos, position_limit=self.position_limits, set_pos=set_pos
        )

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ref_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def run_controller(self):
        """
        Executes one step of the hybrid force-position controller.

        The control flow is:
        1. Update current robot state
        2. For each iteration:
            - Compute compliance error combining pose and wrench errors
            - Apply spatial PD control
            - Transform to appropriate frame if needed
            - Scale the error
            - Convert to joint commands using KDL or direct mapping
        3. Execute using inner controller (joint position, velocity or OSC)

        Returns:
            np.array: Command torques for the robot joints
        """
        # 1. Update state
        self.update()
        # if self.sim.data.time > 0.04:
        #     exit(0)

        if self.use_kdl:
            self.kdl_solver.synchronize_joint_positions(self.joint_pos)

        period = 0.02
        for _ in range(self.iterations):

            net_force, eef_to_base = self.compute_compliance_error()

            # Add virtual force to net force
            net_force += self.virtual_force

            # Compute necessary error terms for PD controller
            cartesian_input = self.compute_spatial_controller(net_force, period)

            if self.frame_of_reference == "eef":
                # convert the error back to the robot_base frame
                cartesian_input = T.rotate_by_transformation(cartesian_input, eef_to_base)

            cartesian_input *= self.error_scale   # scale the entire error here

            if self.use_kdl:
                m_simulated_joint_positions = self.kdl_solver.get_joint_control_cmds(period, cartesian_input)
                if self.inner_controller_type == "JOINT_POSITION":
                    # Inner controller expects relative joint positions
                    self.inner_controller.set_goal(m_simulated_joint_positions - self.joint_pos)
                elif self.inner_controller_type == "OSC_POSE":
                    self.kdl_solver.update_kinematics()
                    m_simulated_pose = self.kdl_solver.get_end_effector_pose()
                    self.inner_controller.set_goal(action=m_simulated_pose)
                else:
                    raise ValueError(f"Invalid inner controller type: {self.inner_controller_type}")
            else:
                self.inner_controller.set_goal(action=cartesian_input)

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Always run superclass call to compute actual torques from desired positions
        return self.inner_controller.run_controller()

    def compute_spatial_controller(self, error, period):
        """
        Implements a PD controller in operational space.

        Args:
            error (np.array): Current 6D error vector (position and orientation)
            period (float): Time period for derivative computation

        Returns:
            np.array: Control output combining proportional and derivative terms
        """
        # Sanitize error to avoid NaNs / Infs propagating through the controller
        if not np.all(np.isfinite(error)):
            error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip pose / wrench error to reasonable range to avoid explosive responses
        error = np.clip(error, -1.0, 1.0)

        # Compute derivative term using the previous ERROR (not previous control output)
        safe_period = max(float(period), 1e-3)
        deriv = (error - self.last_err) / safe_period

        control_error = self.kp * error + self.kd * deriv

        # Clip controller output to avoid generating unreasonably large inner-controller commands
        control_error = np.clip(control_error, -10.0, 10.0)

        # Store the last ERROR for next iteration
        self.last_err = np.copy(error)
        return control_error

    def compute_motion_error(self):
        """
        Computes the pose error between current and desired end-effector state.
        Handles interpolation if enabled.

        Returns:
            np.array: 6D pose error vector [position_error (3), orientation_error (3)]
        """
        desired_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_pos = np.array(self.goal_pos)

        position_error = (desired_pos - self.ref_pos)
        error_norm = np.linalg.norm(position_error)
        if error_norm > 1.0:
            position_error = position_error / error_norm

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ref_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ref_ori_mat)

        # Get rotation angle and axis from orientation error
        ori_angle = np.linalg.norm(ori_error)
        if ori_angle > 1.0:
            # Scale orientation error to have magnitude of 1 radian
            ori_error = ori_error / ori_angle

        # Compute desired force and torque based on errors
        pose_error = np.concatenate([position_error, ori_error])

        return pose_error

    def compute_force_error(self):
        """
        Computes the wrench error between desired and measured forces/torques.

        Returns:
            np.array: 6D wrench error vector [force_error (3), torque_error (3)]
        """
        return self.desired_force_torque - self.eef_wrench

    def compute_compliance_error(self):
        """
        Computes the total compliance error by combining pose and wrench errors according to the selection matrix.
        Handles frame transformations between end-effector and base frames.

        Returns:
            tuple:
                - np.array: Combined 6D compliance error vector
                - np.array or None: Transform from eef to base frame if in eef mode
        """
        pose_error = self.compute_motion_error()

        eef_to_base = None
        if self.frame_of_reference == "eef":
            # Convert pose error to end effector frame
            eef_to_base = self.pose_in_base_from_name(f"{self.ft_prefix}_eef")[:3, :3]
            # Assume that the desired force torque is given in the end effector frame
            pose_error = T.rotate_by_transformation(pose_error, eef_to_base.T)

        elif self.frame_of_reference == "robot_base":
            # Assume that the desired force torque is given in the robot base frame
            pass
        else:
            raise ValueError("Unsupported frame of reference. Only 'eef' and 'robot_base' are supported.")

        wrench_error = self.compute_force_error()
        pose_error_sel = self.selection_matrix * pose_error
        wrench_error_sel = (np.ones_like(self.selection_matrix) - self.selection_matrix) * wrench_error

        # base frame error
        net_force = self.stiffness * pose_error_sel + wrench_error_sel

        return net_force, eef_to_base

    def update_origin(self, origin_pos, origin_ori):
        super().update_origin(origin_pos, origin_ori)
        self.inner_controller.update_origin(origin_pos, origin_ori)

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial configuration
        self.reset_goal()

        self.inner_controller.update_initial_joints(initial_joints)
        self.inner_controller.reset_goal()

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.inner_controller.reset_goal()

        self.goal_ori = np.array(self.ref_ori_mat)
        self.goal_pos = np.array(self.ref_pos)

        self.virtual_force = np.zeros(6)

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ref_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def get_sensor_measurement(self, sensor_name):
        """
        Grabs relevant sensor data from the sim object

        Args:
            sensor_name (str): name of the sensor

        Returns:
            np.array: sensor values
        """
        sensor_idx = np.sum(self.sim.model.sensor_dim[: self.sim.model.sensor_name2id(sensor_name)])
        sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id(sensor_name)]

        return np.array(self.sim.data.sensordata[sensor_idx: sensor_idx + sensor_dim])

    def pose_in_A_to_pose_in_B_by_site_name(self, A, B):
        pos_in_A = self.sim.data.site_xpos[self.sim.model.site_name2id(A)]
        rot_in_A = self.sim.data.site_xmat[self.sim.model.site_name2id(A)].reshape([3, 3])
        pose_in_A = T.make_pose(pos_in_A, rot_in_A)

        pos_in_B = self.sim.data.site_xpos[self.sim.model.site_name2id(B)]
        rot_in_B = self.sim.data.site_xmat[self.sim.model.site_name2id(B)].reshape([3, 3])
        pose_in_B = T.make_pose(pos_in_B, rot_in_B)
        return T.pose_in_A_to_pose_in_B(pose_in_A, pose_in_B)

    @property
    def current_wrench(self):
        return self.wrench_in_base_frame_buf.average

    @property
    def eef_wrench(self):
        return self.wrench_in_eef_frame_buf.average

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.compliance_mode == "variable_stiffness":
            low = np.concatenate([self.input_min, self.force_min, self.torque_min, self.stiffness_min])
            high = np.concatenate([self.input_max, self.force_max,  self.torque_max, self.stiffness_max])
        elif self.compliance_mode == "variable_stiffness_p_gains":
            low = np.concatenate([self.input_min,  self.force_min, self.torque_min, self.stiffness_min, self.kp_min])
            high = np.concatenate([self.input_max, self.force_max, self.torque_max, self.stiffness_max, self.kp_max])
        elif self.compliance_mode == "variable_stiffness_full" or self.compliance_mode == "variable_stiffness_diag_only":
            low = np.concatenate([self.input_min,  self.stiffness_min])
            high = np.concatenate([self.input_max, self.stiffness_max])
        else:  # This is case "fixed"
            low = np.concatenate([self.input_min, self.force_min, self.torque_min])
            high = np.concatenate([self.input_max, self.force_max, self.torque_max])
            # low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return "COMPLIANCE"

    @property
    def input_type(self):
        """Returns the input type for this controller (delta or absolute)"""
        return "delta" if self.use_delta else "absolute"
