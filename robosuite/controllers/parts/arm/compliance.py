from copy import copy
from pathlib import Path
import numpy as np
import rospkg

from robosuite.utils.binding_utils import MjSim
from robosuite.utils.buffers import RingBuffer

from robosuite.utils.sim_utils import compensate_ft_reading
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
COMPLIANCE_MODES = {"fixed", "variable_stiffness", "variable_stiffness_p_gains", "variable_stiffness_full", "variable_stiffness_diag_only"}


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
        sim: MjSim,
        ref_name,
        joint_indexes,
        actuator_range,
        inner_controller_config,
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
        damping_ratio_limits=(0, 100),
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
        self.wrench_in_base_frame_buf = RingBuffer(dim=6, length=ft_buffer_size)
        self.wrench_in_eef_frame_buf = RingBuffer(dim=6, length=ft_buffer_size)
        self.frame_of_reference = frame_of_reference
        self.selection_matrix = selection_matrix
        self.gripper_body_name = gripper_body_name
        if self.gripper_body_name:
            self.gripper_inertial_properties = sim.get_body_inertial_properties(f"{self.ft_prefix}_{gripper_body_name}")

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            lite_physics=lite_physics,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
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
        self.derr_buf = RingBuffer(dim=6, length=2)
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

        if self.use_kdl:
            self.ik_solver = IKSolver(robot='ur5e_powder_grinding_default', rospackage='osx_powder_grinding',
                                      base_link='base_link', ee_link='tool0')
            self.ik_solver.build_generic_model()

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
        gripper_in_robot_base = self.pose_in_base_from_name(f"{self.ft_prefix}_eef")
        wFtS = T.force_frame_transform(gripper_in_robot_base)

        wrench_force = self.get_wrench()

        if self.gripper_body_name:
            wrench_force = compensate_ft_reading(wrench_force[:3], wrench_force[3:],
                                                 self.gripper_inertial_properties['mass'],
                                                 self.gripper_inertial_properties['local_com'],
                                                 self.gripper_inertial_properties['world_rot_mat'],
                                                 self.sim.model._model.opt.gravity)

        current_wrench = np.dot(wFtS, wrench_force)  # compute force/torque reading in base_frame

        self.wrench_in_base_frame_buf.push(current_wrench)
        self.wrench_in_eef_frame_buf.push(wrench_force)

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
            scaled_delta = np.zeros_like(delta)

        # We only want to update goal orientation if there is a valid delta ori value OR if we're using absolute ori
        # use math.isclose instead of numpy because numpy is slow
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

        self.desired_force_torque = desired_ft

    def run_controller(self):
        """
        Executes a hybrid force-position controller that combines position/orientation control with force/torque control.

        The controller operates in either end-effector ('eef') or robot base ('robot_base') frame and uses:
        - Position/orientation control through stiffness-based pose error
        - Force/torque control through wrench error 
        - PD control for dynamic response

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

        Notes:
            - Uses interpolation for smooth position/orientation transitions
            - Selection matrix determines position vs force controlled dimensions
            - Maintains error state between calls for derivative term
            - Can use KDL for inverse kinematics if enabled
        """
        # 1. Update state
        self.update()

        if self.use_kdl:
            self.ik_solver.synchronize_joint_positions(self.joint_pos)

        period = 0.02
        for _ in range(self.iterations):

            net_force, eef_to_base = self.compute_compliance_error()

            # Compute necessary error terms for PD controller
            cartesian_input = self.compute_spatial_controller(net_force, period)

            if self.frame_of_reference == "eef":
                # convert the error back to the robot_base frame
                cartesian_input = self.rotate_by_transformation(cartesian_input, eef_to_base)

            cartesian_input *= self.error_scale  # scale the entire error here

            if self.use_kdl:
                desired_wrench = self.ik_solver.get_joint_control_cmds(period, cartesian_input)
                desired_wrench['positions'] = desired_wrench['positions'] - self.joint_pos
            else:
                desired_wrench = cartesian_input

        # print(f"desired_wrench {desired_wrench}")

        return self.run_inner_controller(desired_wrench)

    def compute_spatial_controller(self, error, period):
        error = self.kp * error + self.kd * (error - self.last_err) / period
        self.last_err = error
        return error

    def compute_motion_error(self):
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
        return self.desired_force_torque - self.eef_wrench

    def compute_compliance_error(self):
        pose_error = self.compute_motion_error()

        eef_to_base = None
        if self.frame_of_reference == "eef":
            # Convert pose error to end effector frame
            eef_to_base = self.pose_in_base_from_name(f"{self.ft_prefix}_eef")[:3, :3]
            # Assume that the desired force torque is given in the end effector frame
            pose_error = self.rotate_by_transformation(pose_error, eef_to_base.T)

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

    def rotate_by_transformation(self, error, A_to_B):
        pos_error = A_to_B @ error[:3]
        ori_error = A_to_B @ error[3:]
        return np.concatenate([pos_error, ori_error])

    def run_inner_controller(self, desired_wrench):
        if self.inner_controller_type == "JOINT_POSITION":
            # Inner controller expects relative joint positions
            self.inner_controller.set_goal(desired_wrench['positions'])
            # position_error = desired_wrench['positions'] - self.joint_pos
            # # vel_pos_error = desired_wrench['velocities'] - self.joint_vel
            # vel_pos_error = - self.joint_vel
            # kp = np.array([10]*6)
            # kd = np.array([0.01]*6)
            # desired_torque = np.multiply(np.array(position_error), kp) + np.multiply(vel_pos_error, kd)

            # # Return desired torques plus gravity compensations
            # self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation
            # return self.torques
        elif self.inner_controller_type == "JOINT_VELOCITY":
            self.inner_controller.set_goal(velocities=desired_wrench)
        elif self.inner_controller_type == "OSC_POSE":
            self.inner_controller.set_goal(action=desired_wrench)

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Always run superclass call to compute actual torques from desired positions
        return self.inner_controller.run_controller()

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

        base_pos_in_world = self.sim.data.get_body_xpos(f"{self.naming_prefix}base")
        base_rot_in_world = self.sim.data.get_body_xmat(f"{self.naming_prefix}base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

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
