from copy import copy
import numpy as np
from robosuite.utils.buffers import RingBuffer

import robosuite.utils.transform_utils as T
from robosuite.controllers.base_controller import Controller
from robosuite.controllers.joint_pos import JointPositionController
from robosuite.controllers.joint_vel import JointVelocityController
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.control_utils import *

# Supported impedance modes
COMPLIANCE_MODES = {"fixed", "variable_stiffness", "variable_stiffness_p_gains", "variable_stiffness_full", "variable_stiffness_diag_only"}
IK_SOLVERS = {"jacobian_transpose", "forward_dynamics"}


class ComplianceController(Controller):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        TODO: add additional docs

    Raises:
        AssertionError: [Invalid compliance mode]
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        inner_controller_config,
        error_scale=1.0,
        stiffness=500,
        kp=0.1,
        kd=0.0,
        compliance_mode="fixed",
        policy_freq=20,
        force_limits=(-50.0, 50.0),
        torque_limits=(-10.0, 10.0),
        ft_buffer_size=10,
        ft_offset=(0., 0., 0., 0., 0., 0.),  # offset payload in world frame
        stiffness_limits=(50, 500),
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        position_limits=None,
        orientation_limits=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_delta=True,
        ik_solver="jacobian_transpose",
        desired_ft_frame="robot_base",  # or "robot_base"
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        self.ft_prefix = eef_name.split('_')[0]
        self.ft_offset = ft_offset
        self.wrench_buf = RingBuffer(dim=6, length=ft_buffer_size)
        self.eef_wrench_buf = RingBuffer(dim=6, length=ft_buffer_size)
        self.desired_ft_frame = desired_ft_frame

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
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
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            **inner_controller_config
        )

        # Make sure that the ik solver is valid
        if self.inner_controller_type == "JOINT_POSITION" or self.inner_controller_type == "JOINT_VELOCITY":
            assert ik_solver in IK_SOLVERS, (
                "Error: Tried to instantiate {} for unsupported "
                "ik solver! Inputted ik solver: {}, Supported solvers: {}".format(self.inner_controller_type, ik_solver, IK_SOLVERS)
            )
        self.ik_solver = ik_solver

        # Verify the proposed impedance mode is supported
        assert compliance_mode in COMPLIANCE_MODES, (
            "Error: Tried to instantiate Compliance controller for unsupported "
            "compliance mode! Inputted compliance mode: {}, Supported modes: {}".format(compliance_mode, COMPLIANCE_MODES)
        )
        self.compliance_mode = compliance_mode

        self.control_dim = 6  # desired position/orientation
        self.input_max = self.nums2array(inner_controller_config['input_max'], self.control_dim)
        self.input_min = self.nums2array(inner_controller_config['input_min'], self.control_dim)
        self.output_max = self.nums2array(inner_controller_config['output_max'], self.control_dim)
        self.output_min = self.nums2array(inner_controller_config['output_min'], self.control_dim)

        self.control_dim += 6  # + force/torque
        self.force_min = self.nums2array(force_limits[0], 3)  # TODO Q: are these imposed anywhere?
        self.force_max = self.nums2array(force_limits[1], 3)
        self.torque_min = self.nums2array(torque_limits[0], 3)
        self.torque_max = self.nums2array(torque_limits[1], 3)

        self.stiffness = self.nums2array(stiffness, 6)
        # stiffness limits
        self.stiffness_min = self.nums2array(stiffness_limits[0], 6)
        self.stiffness_max = self.nums2array(stiffness_limits[1], 6)

        # Add to control dim based on compliance_mode
        if self.compliance_mode == "variable_stiffness":
            self.control_dim += 6
        elif self.compliance_mode == "variable_stiffness_p_gains":
            self.control_dim += 12
        elif self.compliance_mode == "variable_stiffness_diag_only":
            pass
        elif self.compliance_mode == "variable_stiffness_full":
            self.control_dim = 18

            self.stiffness = self.nums2array(stiffness, 12)
            # stiffness limits
            self.stiffness_min = self.nums2array(stiffness_limits[0], 12)
            self.stiffness_max = self.nums2array(stiffness_limits[1], 12)

        self.use_delta = control_delta

        self.kp = self.nums2array(kp, 6)
        self.kd = self.nums2array(kd, 6)
        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        self.error_scale = error_scale

        self.last_err = np.zeros(6)
        self.derr_buf = RingBuffer(dim=6, length=5)
        self.last_joint_vel = np.zeros(6)

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

    def update(self):
        super().update()

        # Synchronize Joint Positions
        self.current_joint_positions = self.joint_pos
        self.last_joint_positions = copy(self.joint_pos)
        self.current_joint_velocities = self.joint_vel
        self.last_joint_velocities = copy(self.joint_vel)

        self.transform_wrench_to_base_frame()

    def get_wrench(self):
        return np.concatenate([
            self.get_sensor_measurement(f"{self.ft_prefix}_force_ee"),
            self.get_sensor_measurement(f"{self.ft_prefix}_torque_ee"),
        ])

    def transform_wrench_to_base_frame(self):
        # Compute force/torque
        # get sensor f/t measurements from gripper site, transform to world frame
        wrench_force = self.get_wrench()
        wrench_force -= self.ft_offset

        gripper_in_robot_base = self.pose_in_base_from_name(f"{self.ft_prefix}_eef")
        wFtS = T.force_frame_transform(gripper_in_robot_base)
        current_wrench = np.dot(wFtS, wrench_force)

        self.wrench_buf.push(current_wrench)
        self.eef_wrench_buf.push(wrench_force)

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command] # TODO change name in docs to eef pose
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # Update state
        self.update()

        if self.compliance_mode == "variable_stiffness":
            delta, desired_ft, stiffness = action[:6], action[6:12], action[12:]
            self.stiffness = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
        elif self.compliance_mode == "variable_stiffness_p_gains":
            delta, desired_ft, stiffness, kp = action[:6], action[6:12], action[12:18], action[18:]
            self.stiffness = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
        elif self.compliance_mode == "variable_stiffness_diag_only":
            stiffness, delta = action[:6], action[6:]
            self.stiffness = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
            desired_ft = np.zeros(6)
        elif self.compliance_mode == "variable_stiffness_full":
            cholesky_stiffness, delta = action[:12], action[12:]
            stiffness_pos_matrix = T.cholesky_vector_to_spd(cholesky_stiffness[:6])
            stiffness_ori_matrix = T.cholesky_vector_to_spd(cholesky_stiffness[6:])
            stiffness = np.concatenate([stiffness_pos_matrix.flatten(), stiffness_ori_matrix.flatten()])

            self.stiffness = np.zeros_like(stiffness)

            # assume positive diagonal stiffness
            diag_indices = [0, 4, 8, 9, 13, 17]
            self.stiffness[diag_indices] = np.clip(stiffness[diag_indices], self.stiffness_min[0], self.stiffness_max[0])
            # other values have no min value, it can even be negative up to the -stiffness_max value
            other_indices = np.ones(len(stiffness), bool)
            other_indices[diag_indices] = False
            self.stiffness[other_indices] = np.sign(stiffness[other_indices]) * np.clip(np.abs(stiffness[other_indices]), 0, self.stiffness_max[0])

            # TODO:(cambel) Here we only use the diagonal values
            self.stiffness = self.stiffness[diag_indices]
            desired_ft = np.zeros(6)
        else:  # This is case "fixed"
            delta, desired_ft = action[:6], action[6:]
        desired_ft[:3] = np.clip(desired_ft[:3], self.force_min, self.force_max)
        desired_ft[3:] = np.clip(desired_ft[3:], self.torque_min, self.torque_max)

        # If we're using deltas, interpret actions as such
        if self.use_delta:
            if delta is not None:
                scaled_delta = self.scale_action(delta)
            else:
                scaled_delta = []
        # Else, interpret actions as absolute values
        else:
            if set_pos is None:
                set_pos = delta[:3]
            if set_ori is None:
                set_ori = (T.quat2mat(T.axisangle2quat(delta[3:6])))
            # No scaling of values since these are absolute values
            scaled_delta = delta

        # We only want to update goal orientation if there is a valid delta ori value OR if we're using absolute ori
        # use math.isclose instead of numpy because numpy is slow
        self.goal_ori = set_goal_orientation(
            scaled_delta[3:], self.ee_ori_mat, orientation_limit=self.orientation_limits, set_ori=set_ori
        )
        self.goal_pos = set_goal_position(
            scaled_delta[:3], self.ee_pos, position_limit=self.position_limits, set_pos=set_pos
        )

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        self.desired_force_torque = desired_ft

    def run_controller(self):
        """
        TODO: docs

        Returns:
             np.array: Command torques
        """

        # 1. Update state
        self.update()

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

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        # Compute desired force and torque based on errors
        position_error = (desired_pos - self.ee_pos)
        # TODO (cambel): force/torque is too sensitive, how to fix that?
        force_torque_error = (self.desired_force_torque - self.current_wrench) * [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
        pose_error = np.concatenate([position_error, ori_error])

        # apply the selection matrix in gripper frame
        selection_matrix_gripper_frame = np.diag([1, 1, 1, 1, 1, 1])
        eef_to_base = self.pose_in_base_from_name(f"{self.ft_prefix}_eef")[:3, :3]  # get just rotation matrix
        pose_error_sel, force_torque_error_sel = self.apply_selection_matrix_gripper_frame(pose_error, force_torque_error, selection_matrix_gripper_frame, eef_to_base)

        # base frame error
        error = self.stiffness * pose_error_sel + force_torque_error_sel

        # Compute necessary error terms for PD controller
        derr = error - self.last_err
        self.last_err = error
        self.derr_buf.push(derr)
        spatial_controller = self.kp * error + self.kd * self.derr_buf.average / self.period

        desired_wrench = self.error_scale * spatial_controller
        # print("desired_wrench", desired_wrench)

        if self.inner_controller_type == "JOINT_POSITION" \
                or self.inner_controller_type == "JOINT_VELOCITY":
            return self.use_joint_pos_vel(desired_wrench)
        elif self.inner_controller_type == "OSC_POSE":
            return self.use_osc(desired_wrench)

    def apply_selection_matrix_gripper_frame(self, pos_error_ref, force_error_ref, selection_matrix_gripper, R_gripper_to_ref):  # TODO maybe with quat is faster
        # Convert errors from reference frame to gripper frame
        pos_error_gripper = R_gripper_to_ref.T @ pos_error_ref[:3]  # transpose,not inv since its just rotation, without translation
        ori_error_gripper = R_gripper_to_ref.T @ pos_error_ref[3:]
        force_error_gripper = R_gripper_to_ref.T @ force_error_ref[:3]
        torque_error_gripper = R_gripper_to_ref.T @ force_error_ref[3:]

        # Apply selection matrix in gripper frame
        selected_pose_error_gripper = selection_matrix_gripper @ np.concatenate([pos_error_gripper, ori_error_gripper])
        selected_ft_error_gripper = (np.eye(6) - selection_matrix_gripper) @ np.concatenate([force_error_gripper, torque_error_gripper])

        # Convert selected errors back to reference frame
        selected_pos_error_ref = R_gripper_to_ref @ selected_pose_error_gripper[:3]
        selected_ori_error_ref = R_gripper_to_ref @ selected_pose_error_gripper[3:]

        selected_force_error_ref = R_gripper_to_ref @ selected_ft_error_gripper[:3]
        selected_torque_error_ref = R_gripper_to_ref @ selected_ft_error_gripper[3:]

        return np.concatenate([selected_pos_error_ref, selected_ori_error_ref]), np.concatenate([selected_force_error_ref, selected_torque_error_ref])

    def use_osc(self, desired_wrench):
        self.inner_controller.set_goal(action=desired_wrench)
        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Always run superclass call to compute actual torques from desired positions
        return self.inner_controller.run_controller()

    def use_joint_pos_vel(self, desired_wrench):
        # TODO: fix, this does not quite work

        # 2. compute the net force
        # for-loop of iterations at some internal_period
        # computeComplianceError() -> net_force = stiffness[m_compliance_ref_link] @ computeMotionError() + computeForceError()
        # Optionally include selection_matrix

        # 3. computeJointControlCmds(error, period)
        # m_cartesian_input = m_error_scale * m_spatial_controller(error, period)
        # m_simulated_joint_motion = getJointControlCmds(period, m_cartesian_input)
        # buildGenericModel()
        # mass_matrix, jacobian
        # Compute joint accelerations according to: \f$ \ddot{q} = H^{-1} ( J^T f) \f$
        # current_acceleration = inertia.inverse * jacobian.transpose * net_force
        # current_positions = last_positions + last_velocities * period
        # current_velocities = last_velocities + current_accelerations * period
        # current_velocities *= 0.9

        # 4. write final commands to hardware interface

        if self.ik_solver == "jacobian_transpose":
            self.compute_jacobian_transpose(desired_wrench)
        elif self.ik_solver == "forward_dynamics":
            self.compute_forward_dynamics(desired_wrench)

        # print("goal_vel", self.period)

        if self.inner_controller_type == "JOINT_POSITION":
            self.inner_controller.set_goal(action=self.current_joint_positions)
        else:
            self.inner_controller.set_goal(velocities=self.current_joint_velocities)

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        self.last_joint_velocities = copy(self.current_joint_velocities)
        self.last_joint_positions = copy(self.current_joint_positions)

        # Always run superclass call to compute actual torques from desired positions
        return self.inner_controller.run_controller()

    def compute_forward_dynamics(self, desired_wrench):
        # Compute joint accelerations according to: \f$ \ddot{q} = H^{-1} ( J^T f) \f$
        self.current_joint_accelerations = np.linalg.inv(self.mass_matrix) @ self.J_full.T @ desired_wrench
        self.current_joint_positions = self.last_joint_positions + self.last_joint_vel * self.period
        self.current_joint_velocities = self.last_joint_velocities + self.current_joint_accelerations * self.period
        self.current_joint_velocities *= 0.9

    def compute_jacobian_transpose(self, desired_wrench):
        # Compute joint accelerations according to: \f$ \ddot{q} = ( J^T f) \f$
        # self.current_joint_accelerations = np.linalg.inv(self.J_full) @ desired_wrench # cartesian error to joint error
        mass_matrix_inv = np.linalg.inv(self.mass_matrix)
        lambda_full = np.dot(np.dot(self.J_full, mass_matrix_inv), self.J_full.transpose())
        self.current_joint_accelerations = np.linalg.inv(self.mass_matrix) @ self.J_full.T @ desired_wrench
        # self.current_joint_accelerations = np.dot(lambda_full, desired_wrench)
        self.current_joint_velocities = self.last_joint_velocities + 0.5 * self.current_joint_accelerations * self.period
        # self.current_joint_positions = self.last_joint_positions + 0.5 * self.current_joint_velocities * self.period

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

        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    @property
    def current_wrench(self):
        return self.wrench_buf.average

    @ property
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

    @ property
    def name(self):
        return "COMPLIANCE"

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

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.

        Args:
            name (str): Name of body in sim to grab pose

        Returns:
            np.array: (4,4) array corresponding to the pose of @name in the base frame
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos(f"robot{self.ft_prefix[-1]}_base")
        base_rot_in_world = self.sim.data.get_body_xmat(f"robot{self.ft_prefix[-1]}_base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base
