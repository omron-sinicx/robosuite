import math
import numpy as np
from scipy.signal import butter, filtfilt
from robosuite.utils.buffers import DeltaBuffer

import robosuite.utils.transform_utils as T
from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp", "variable_full_kp"}

# TODO: Maybe better naming scheme to differentiate between input / output min / max and pos/ori limits, etc.
# TODO: if completely functional, merge with OSC itself? ! For now just works just with fixed


class OperationalSpaceControllerFT(Controller):
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

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        control_delta (bool): Whether to control the robot using delta or absolute commands (where absolute commands
            are taken in the world coordinate frame)

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        ft_ref_flag (bool): if set to true, action space has 6 additional elements corresponding to 
            forces on x y z and torques on x y z in the operational frame

        ft_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @ft_ref_flag is set to either
            "true". Sets the corresponding min / max ranges of the forces on x y z and torques on x y z in the operational space frame

        force_active_case (str): determines what the controller does with the calculated active force. If "both" the wrench
            will be the sum of wrench from position controller and active force controller; if "active" the position controller wrench is ignored,
            leading to only direct force control; if "hybrid" the @selection_matrix will be used to select which axes are for force control and
            which for position control; any other string will lead to just position control, the original OSC

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        kp=150,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_ori=True,
        control_delta=True,
        uncouple_pos_ori=True,
        ft_ref_flag=True,
        ft_limits=(0, 20),
        force_active_case="position",
        kp_force=np.array([10., 10., 10., 10., 10., 10.]),
        ki_force=np.array([1., 1., 1., 1., 1., 1.]),
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        self.ft_prefix = eef_name.split('_')[0]

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        # Control dimension
        self.control_dim = 6 if self.use_ori else 3
        self.name_suffix = "POSE" if self.use_ori else "POSITION"

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # desired force\torque [fx fy fz tx ty tz] in operational frame and their limits
        self.FT_reference = np.zeros(6)
        self.ft_ref_flag = ft_ref_flag
        self.force_active_case = force_active_case if self.ft_ref_flag == True else "position"
        self.ft_min = self.nums2array(ft_limits[0], 6)
        self.ft_max = self.nums2array(ft_limits[1], 6)
        self.ee_ft = DeltaBuffer(dim=6)  # current and last values recorded for force/torque at eef
        self.F_active = DeltaBuffer(dim=6)  # current and last values just for active force

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, (
            "Error: Tried to instantiate OSC controller for unsupported "
            "impedance mode! Inputted impedance mode: {}, Supported modes: {}".format(impedance_mode, IMPEDANCE_MODES)
        )

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim += 12
        elif self.impedance_mode == "variable_kp":
            self.control_dim += 6
        elif self.impedance_mode == "variable_full_kp":
            self.control_dim += 18
            self.kp_min = self.nums2array(kp_limits[0], 18)
            self.kp_max = self.nums2array(kp_limits[1], 18)

        if self.ft_ref_flag == True:
            self.control_dim += 6

        # limits
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals based on initial pos / ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)

        self.relative_ori = np.zeros(3)
        self.ori_ref = None

        self.integral = 0
        self.last_error = 0

        self.kp_force = kp_force
        self.ki_force = ki_force

        # Default to position control
        self.selection_matrix = np.eye(6)
        self.selection_matrix /= 2.0

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary TODO make this cleaner
        if self.impedance_mode == "variable":
            if self.ft_ref_flag is True:
                damping_ratio, kp, delta, self.FT_reference = action[:6], action[6:12], action[12:18], action[18:]
            else:
                damping_ratio, kp, delta = action[:6], action[6:12], action[12:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            if self.ft_ref_flag is True:
                kp, delta, self.FT_reference = action[:6], action[6:12], action[12:]
            else:
                kp, delta = action[:6], action[6:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        elif self.impedance_mode == "variable_full_kp":
            kp, delta = action[:18], action[18:]
            self.kp = np.zeros_like(kp)
            # assume positive diagonal stiffness
            diag_indices = [0, 4, 8, 9, 13, 17]
            self.kp[diag_indices] = np.clip(kp[diag_indices], self.kp_min[diag_indices], self.kp_max[diag_indices])
            # other values have no min value, it can even be negative up to the -kp_max value
            other_indices = np.ones(len(kp), np.bool)
            other_indices[diag_indices] = False
            self.kp[other_indices] = np.sign(kp[other_indices]) * np.clip(np.abs(kp[other_indices]), 0, self.kp_max[other_indices])
            # Compute damping (preserve sign)
            self.kd = 2 * np.sign(self.kp) * np.sqrt(np.abs(self.kp))  # critically damped
        else:  # This is case "fixed"
            if self.ft_ref_flag is True:
                delta, self.FT_reference = action[:6], action[6:]
            else:
                delta = action

        # If we're using deltas, interpret actions as such
        if self.use_delta:
            if delta is not None:
                scaled_delta = self.scale_action(delta)
                if not self.use_ori and set_ori is None:
                    # Set default control for ori since user isn't actively controlling ori
                    set_ori = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            else:
                scaled_delta = []
        # Else, interpret actions as absolute values
        else:
            if set_pos is None:
                set_pos = delta[:3]
            # Set default control for ori if we're only using position control
            if set_ori is None:
                set_ori = (
                    T.quat2mat(T.axisangle2quat(delta[3:6]))
                    if self.use_ori
                    else np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
                )
            # No scaling of values since these are absolute values
            scaled_delta = delta

        # We only want to update goal orientation if there is a valid delta ori value OR if we're using absolute ori
        # use math.isclose instead of numpy because numpy is slow
        bools = [0.0 if math.isclose(elem, 0.0) else 1.0 for elem in scaled_delta[3:]]
        if sum(bools) > 0.0 or set_ori is not None:
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

    def update(self, force=False):
        # Override to compute F/T
        super().update(force=force)

        # get sensor f/t measurements from gripper site, transform to world frame
        offset = np.array([0.0011635382606757878, -0.0011644652330827126, 2.9429995396193043, 0.000520645845461271, -0.00042506640050181784, -3.740287389696474e-07])  # offset payload in world frame

        gripper_in_world = self.pose_in_world_from_name(f"{self.ft_prefix}_eef")
        ee_force, ee_torque = T.force_in_A_to_force_in_B(self.get_sensor_measurement(f"{self.ft_prefix}_force_ee"),
                                                         self.get_sensor_measurement(f"{self.ft_prefix}_torque_ee"),
                                                         gripper_in_world)
        self.current_wrench = np.concatenate([ee_force, ee_torque])
        self.current_wrench -= offset

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
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
        position_error = desired_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel

        if self.impedance_mode != "variable_full_kp":
            position_kp = np.diag(self.kp[0:3])
            orientation_kp = np.diag(self.kp[3:6])
        else:
            position_kp = np.array(self.kp[0:9]).reshape((3, 3))
            orientation_kp = np.array(self.kp[9:18]).reshape((3, 3))

        # filter ft measurements
        filtered_wrench = self.butterworth_filter(self.ee_ft, self.current_wrench, 2)

        self.ee_ft.push(filtered_wrench)
        force_error = self.FT_reference - self.ee_ft.current

        # Fm
        desired_force = np.dot(position_error, position_kp) + np.multiply(vel_pos_error, self.kd[0:3])

        vel_ori_error = -self.ee_ori_vel

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.dot(ori_error, orientation_kp) + np.multiply(vel_ori_error, self.kd[3:6])

        # TODO clip forces
        F_active = self.PID(error=force_error, kp=self.kp_force, ki=self.ki_force, kd=np.zeros(6))
        self.F_active.push(F_active)

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)  # lambda * Fm

        if self.force_active_case == "active":
            # F = Fa
            decoupled_wrench = self.F_active.current
        elif self.force_active_case == "both":
            # F = lambda*Fm + Fa
            decoupled_wrench += self.F_active.current
        elif self.force_active_case == "hybrid":
            # F = S*(lambda*Fm) + (1-S)*Fa
            decoupled_wrench = self.selection_matrix @ decoupled_wrench \
                + (np.eye(6)-self.selection_matrix) @ self.F_active.current
        elif self.force_active_case == "position":  # case no-active, just position
            # F = lambda*Fm
            pass
        else:
            raise ValueError(f"Unsupported method: {self.force_active_case}. Valid methods: active, both, hybrid, position")

        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = self.J_full.T @ decoupled_wrench + self.torque_compensation

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )
        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial configuration
        self.reset_goal()

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
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
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_full_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max

        if self.ft_ref_flag == True:
            low = np.concatenate([low, self.ft_min])
            high = np.concatenate([high, self.ft_max])

        return low, high

    @property
    def name(self):
        return "OSC_" + self.name_suffix + "_FT"

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

        base_pos_in_world = self.sim.data.get_body_xpos("robot0_base")
        base_rot_in_world = self.sim.data.get_body_xmat("robot0_base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def pose_in_world_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the world frame.

        Args:
            name (str): Name of body in sim to grab pose

        Returns:
            np.array: (4,4) array corresponding to the pose of @name in the world frame
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        return pose_in_world

    def PID(self, error, kp, ki=None, kd=None):

        windup = (np.ones_like(kp))
        i_min = -np.array(windup)
        i_max = np.array(windup)

        dt = 1.0 / self.control_freq
        delta_error = error - self.last_error

        # Compute terms
        self.integral += error * dt
        p_term = kp * error
        i_term = ki * self.integral
        i_term = np.maximum(i_min, np.minimum(i_term, i_max))

        # First delta error is huge since it was initialized at zero first, avoid considering
        if not np.allclose(self.last_error, np.zeros_like(self.last_error)):
            d_term = kd * delta_error / dt
        else:
            d_term = kd * np.zeros_like(delta_error) / dt

        output = p_term + i_term + d_term
        # Save last values
        self.last_error = np.array(error)
        return output

    def simple_moving_average_filter(self, buffer, current_measurement):
        """
        Uses current measurements to update buffer of measurements

        Args:
            buffer (DeltaBuffer): buffer of measurements
            current_measurement (np.array): array of measurements

        Returns:
            filtered measurement
        """

        # Calculate the average of current window
        filtered = (buffer.last + buffer.current + current_measurement) / 3

        return filtered

    def butterworth_filter(self, buffer, current_measurement, w):
        """
        Uses current measurements to update buffer of measurements with a butterworth filter

        Args:
            buffer (DeltaBuffer): buffer of measurements
            current_measurement (np.array): array of measurements
            w (scalar): cutoff frequency of the filter, 0 < wn < fs/2

        Returns:
            filtered measurement
        """
        fsc = self.control_freq
        b, a = butter(5, w, 'low', fs=fsc)
        signal = np.concatenate([buffer.last, buffer.current, current_measurement]).reshape(3, 6)
        return filtfilt(b, a, signal, axis=0, padlen=0)[2, :]
