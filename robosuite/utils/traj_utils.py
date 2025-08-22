from scipy.interpolate import CubicSpline
from dataclasses import dataclass
import abc

import numpy as np
import quaternion

import robosuite.utils.transform_utils as T


# Classes for trajectory interpolation
class Interpolator(object, metaclass=abc.ABCMeta):
    """
    General interpolator interface.
    """

    @abc.abstractmethod
    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        Returns:
            np.array: Next interpolated step
        """
        raise NotImplementedError


class LinearInterpolator(Interpolator):
    """
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        ndim (int): Number of dimensions to interpolate

        controller_freq (float): Frequency (Hz) of the controller

        policy_freq (float): Frequency (Hz) of the policy model

        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.

            :Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update

        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:

                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(
        self,
        ndim,
        controller_freq,
        policy_freq,
        ramp_ratio=0.2,
        use_delta_goal=False,
        ori_interpolate=None,
    ):
        self.dim = ndim  # Number of dimensions to interpolate
        self.ori_interpolate = ori_interpolate  # Whether this is interpolating orientation or not
        self.order = 1  # Order of the interpolator (1 = linear)
        self.step = 0  # Current step of the interpolator
        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # Total num steps per interpolator action
        self.use_delta_goal = use_delta_goal  # Whether to use delta or absolute goals (currently
        # not implemented yet- TODO)
        self.set_states(dim=ndim, ori=ori_interpolate)

    def set_states(self, dim=None, ori=None):
        """
        Updates self.dim and self.ori_interpolate.

        Initializes self.start and self.goal with correct dimensions.

        Args:
            ndim (None or int): Number of dimensions to interpolate

            ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
                Specified string determines assumed type of input:

                    `'euler'`: Euler orientation inputs
                    `'quat'`: Quaternion inputs
        """
        # Update self.dim and self.ori_interpolate
        self.dim = dim if dim is not None else self.dim
        self.ori_interpolate = ori if ori is not None else self.ori_interpolate

        # Set start and goal states
        if self.ori_interpolate is not None:
            if self.ori_interpolate == "euler":
                self.start = np.zeros(3)
            else:  # quaternions
                self.start = np.array((0, 0, 0, 1))
        else:
            self.start = np.zeros(self.dim)
        self.goal = np.array(self.start)

    def set_goal(self, goal):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step

        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError(
                "LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(goal.shape[0], self.dim)
            )

        # Update start and goal
        self.start = np.array(self.goal)
        self.goal = np.array(goal)

        # Reset interpolation steps
        self.step = 0

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """
        # Grab start position
        x = np.array(self.start)
        # Calculate the desired next step based on remaining interpolation steps
        if self.ori_interpolate is not None:
            # This is an orientation interpolation, so we interpolate linearly around a sphere instead
            goal = np.array(self.goal)
            if self.ori_interpolate == "euler":
                # this is assumed to be euler angles (x,y,z), so we need to first map to quat
                x = T.mat2quat(T.euler2mat(x))
                goal = T.mat2quat(T.euler2mat(self.goal))

            # Interpolate to the next sequence
            x_current = T.quat_slerp(x, goal, fraction=(self.step + 1) / self.total_steps)
            if self.ori_interpolate == "euler":
                # Map back to euler
                x_current = T.mat2euler(T.quat2mat(x_current))
        else:
            # This is a normal interpolation
            dx = (self.goal - x) / (self.total_steps - self.step)
            x_current = x + dx

        # Increment step if there's still steps remaining based on ramp ratio
        if self.step < self.total_steps - 1:
            self.step += 1

        # Return the new interpolated step
        return x_current


def _generate_mortar_trajectory_core(mortar_diameter, desired_height, n_steps, default_quat=np.array([0, -1, 0, 0]), fraction=None, max_angle=None, pestle_radius=0.0125, circumferential_offset=0.0):
    """
    Core logic for generating a mortar trajectory. This function contains the common
    trajectory generation logic used by both step-based and time-based trajectory functions.

    Args:
        mortar_diameter (float): Diameter of the bowl in meters
        desired_height (float): Desired height from the bottom of the bowl in meters
        n_steps (int): Number of points in the trajectory
        default_quat (list): Quaternion [qx, qy, qz, qw] representing orientation at the bowl center at (0,0,0)
        fraction: (float): If defined, the final quaternion returned is the slerp fraction from the default_quat to the 
                           corresponding normal vector
        max_angle (float): If defined, takes priority over fraction. Maximum angle (in radians) between default_quat
                           and the final quaternion. The fraction will be calculated to ensure this constraint.
        pestle_radius (float): Radius of the pestle tip in meters. Used to adjust trajectory to prevent penetration
                              when inclination is constrained.
        circumferential_offset (float): Offset in radians for initial orientation of the trajectory
    Returns:
        np.array: Array of shape (n_steps, 7) containing [x, y, z, qx, qy, qz, qw]
                 for each point in the trajectory
    """
    # Step 1: Calculate bowl parameters
    radius = mortar_diameter / 2

    # Step 2: Verify if desired height is valid
    if desired_height > radius:
        raise ValueError("Desired height cannot be greater than bowl radius")

    # Step 3: Calculate radius of the circle at desired height
    # For upward facing bowl: r^2 = R^2 - (R-h)^2
    circle_radius = np.sqrt(radius**2 - (radius - desired_height)**2)

    # Step 4: Generate points along a circle at desired height
    theta = np.linspace(circumferential_offset, 2*np.pi + circumferential_offset, n_steps)

    # Step 5: Calculate normal vectors at each point on the original circle
    # For an upward facing bowl, the normal vector points outward from the center of curvature
    normals = np.zeros((n_steps, 3))

    # First, calculate normals at the original circle points to determine inclination constraints
    original_x = circle_radius * np.cos(theta)
    original_y = circle_radius * np.sin(theta)
    original_z = np.full_like(theta, desired_height)

    for i in range(n_steps):
        point = np.array([original_x[i], original_y[i], original_z[i] - radius])  # Shift center of curvature to (0,0,-R)
        normal = -point / np.linalg.norm(point)  # Normalize and negate for outward normal
        normals[i] = normal

    # Step 6: Calculate quaternions and determine penetration adjustment
    quaternions = np.zeros((n_steps, 4))
    initial_rotation = T.quat2mat(default_quat)

    # Calculate the maximum penetration depth to adjust circle radius
    max_penetration = 0.0

    for i in range(n_steps):
        normal = normals[i]

        # Calculate rotation from [0, 0, 1] to normal vector
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, normal)
        s = np.linalg.norm(v)

        if s < 1e-10:  # If vectors are parallel
            if normal[2] > 0:  # Same direction
                surface_rotation = T.quat2mat([0, 0, 0, 1])
            else:  # Opposite direction
                surface_rotation = T.quat2mat([1, 0, 0, 0])
        else:
            c = np.dot(z_axis, normal)
            v_skew = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
            R_matrix = np.eye(3) + v_skew + np.matmul(v_skew, v_skew) * (1 - c) / (s * s)
            surface_rotation = R_matrix

        # Compose rotations: first apply initial orientation, then surface normal rotation
        final_rotation = surface_rotation @ initial_rotation
        final_quaternion = T.mat2quat(final_rotation)

        # Store the unconstrained quaternion to calculate penetration
        unconstrained_quaternion = np.array(final_quaternion)

        # Apply constraints if specified
        if max_angle is not None:
            total_angle = compute_quat_angle(default_quat, final_quaternion)
            # If the angle is greater than max_angle, calculate the appropriate fraction
            if total_angle > max_angle:
                # Calculate the fraction that would give us max_angle
                computed_fraction = max_angle / total_angle
                final_quaternion = T.quat_slerp(default_quat, final_quaternion, fraction=computed_fraction)

                # Calculate penetration due to constrained inclination
                if pestle_radius > 0:
                    # Calculate the difference between ideal and constrained orientation
                    angle_diff = total_angle - max_angle
                    # Estimate penetration based on pestle radius and angle difference
                    # Using trigonometry: penetration ≈ pestle_radius * (1 - cos(angle_diff))
                    penetration = pestle_radius * (1 - np.cos(angle_diff))
                    max_penetration = max(max_penetration, penetration)
        elif fraction is not None:
            # Store the unconstrained quaternion
            unconstrained_quaternion = np.array(final_quaternion)

            # Apply the fraction constraint
            final_quaternion = T.quat_slerp(default_quat, final_quaternion, fraction=fraction)

            # Calculate penetration due to constrained inclination
            if pestle_radius > 0:
                # Calculate the angle between unconstrained and constrained orientation
                angle_diff = compute_quat_angle(final_quaternion, unconstrained_quaternion)
                # Estimate penetration based on pestle radius and angle difference
                penetration = pestle_radius * (1 - np.cos(angle_diff))
                max_penetration = max(max_penetration, penetration)

        quaternions[i] = final_quaternion

    # Adjust circle radius to prevent penetration
    adjusted_circle_radius = circle_radius
    if pestle_radius > 0 and max_penetration > 0:
        # Reduce the circle radius by the maximum penetration depth
        adjusted_circle_radius = max(0, circle_radius - max_penetration)

    # Generate adjusted trajectory points
    x = adjusted_circle_radius * np.cos(theta)
    y = adjusted_circle_radius * np.sin(theta)
    z = np.full_like(theta, desired_height)

    # Step 7: Combine positions and orientations
    trajectory = np.column_stack((x, y, z, quaternions))

    return trajectory


def generate_mortar_trajectory(mortar_diameter, desired_height, n_steps, default_quat=np.array([0, -1, 0, 0]), fraction=None, max_angle=None, pestle_radius=0.0125):
    """
    Generate a trajectory to trace the surface of an upward-facing bowl at a given height.
    The pen orientation at the center (0,0,0) is represented by quaternion [0,-1,0,0].

    Args:
        mortar_diameter (float): Diameter of the bowl in meters
        desired_height (float): Desired height from the bottom of the bowl in meters
        n_steps (int): Number of points in the trajectory
        default_quat (list): Quaternion [qx, qy, qz, qw] representing orientation at the bowl center at (0,0,0)
        fraction: (float): If defined, the final quaternion returned is the slerp fraction from the default_quat to the 
                           corresponding normal vector
        max_angle (float): If defined, takes priority over fraction. Maximum angle (in radians) between default_quat
                           and the final quaternion. The fraction will be calculated to ensure this constraint.
        pestle_radius (float): Radius of the pestle tip in meters. Used to adjust trajectory to prevent penetration
                              when inclination is constrained.

    Returns:
        np.array: Array of shape (n_steps, 7) containing [x, y, z, qx, qy, qz, qw]
                 for each point in the trajectory
    """
    trajectory = _generate_mortar_trajectory_core(
        mortar_diameter, desired_height, n_steps, default_quat,
        fraction, max_angle, pestle_radius
    )

    # Add initial pose to the end of the trajectory to complete the circle
    trajectory = np.concatenate([trajectory, [trajectory[0]]])

    return trajectory


def generate_mortar_trajectory_timed(
        mortar_diameter, desired_height, control_frequency, duration, total_timesteps, default_quat=np.array([0, -1, 0, 0]),
        fraction=None, max_angle=None, pestle_radius=0.0125, circumferential_offset=0.0):
    """
    Generate a time-based trajectory to trace the surface of an upward-facing bowl at a given height.
    The trajectory duration and number of revolutions are determined by the control frequency, 
    revolution duration, and total timesteps.

    Args:
        mortar_diameter (float): Diameter of the bowl in meters
        desired_height (float): Desired height from the bottom of the bowl in meters
        control_frequency (float): Control frequency in Hz
        duration (float): Time in seconds for one complete revolution
        total_timesteps (int): Total number of timesteps for the trajectory
        default_quat (list): Quaternion [qx, qy, qz, qw] representing orientation at the bowl center at (0,0,0)
        fraction: (float): If defined, the final quaternion returned is the slerp fraction from the default_quat to the 
                           corresponding normal vector
        max_angle (float): If defined, takes priority over fraction. Maximum angle (in radians) between default_quat
                           and the final quaternion. The fraction will be calculated to ensure this constraint.
        pestle_radius (float): Radius of the pestle tip in meters. Used to adjust trajectory to prevent penetration
                              when inclination is constrained.
        circumferential_offset (float): Offset in radians for initial orientation of the trajectory

    Returns:
        np.array: Array of shape (total_timesteps, 7) containing [x, y, z, qx, qy, qz, qw]
                 for each point in the trajectory
    """
    # Calculate timesteps per revolution
    timesteps_per_revolution = int(control_frequency * duration)

    # Calculate how many full revolutions and remainder timesteps
    full_revolutions = total_timesteps // timesteps_per_revolution
    remainder_timesteps = total_timesteps % timesteps_per_revolution

    # Generate a single revolution trajectory
    single_revolution_trajectory = _generate_mortar_trajectory_core(
        mortar_diameter, desired_height, timesteps_per_revolution,
        default_quat, fraction, max_angle, pestle_radius, circumferential_offset
    )

    # Build the complete trajectory
    trajectory_points = []

    # Add full revolutions
    for _ in range(full_revolutions):
        trajectory_points.extend(single_revolution_trajectory)

    # Add partial revolution if there are remainder timesteps
    if remainder_timesteps > 0:
        trajectory_points.extend(single_revolution_trajectory[:remainder_timesteps])

    # Convert to numpy array
    trajectory = np.array(trajectory_points)

    assert len(trajectory) == total_timesteps, f"Trajectory length {len(trajectory)} does not match total_timesteps {total_timesteps}"

    return trajectory


def compute_quat_angle(quat1, quat2):
    # Calculate the angle between default_quat and final_quaternion
    dot = np.dot(quat1, quat2)
    # Ensure we're taking the shortest path
    if dot < 0:
        quat2 = -quat2
        dot = -dot
    # Clamp dot product for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    # Calculate the total angle between quaternions
    total_angle = np.arccos(dot) * 2
    return total_angle


def compute_max_step_size(trajectory):
    """
    Computes the maximum step size for each dimension in a trajectory.
    For trajectories with both position and orientation (7D: xyz + quaternion),
    position dimensions are computed directly, while orientation uses quaternion
    error calculation.

    Args:
        trajectory (np.ndarray): Array of shape (N, M) containing N waypoints of 
                               M dimensions each. For pos+quat trajectories, 
                               M should be 7 (3 for position, 4 for quaternion).

    Returns:
        np.ndarray: For translation-only trajectories, returns array of shape (M,)
                   containing the maximum step size for each dimension.
                   For trajectories with orientation (7D), returns array of shape (6,)
                   containing max step size for position (3) and orientation error (3).
    """
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)

    if len(trajectory.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {trajectory.shape}")

    # Check if this is a position+orientation trajectory (should be 7D)
    if trajectory.shape[1] == 7:  # xyz + quaternion
        # Initialize max step sizes array for 6D result (3 position + 3 orientation)
        max_step_sizes = np.zeros(6)

        # For position dimensions (first 3), calculate as before
        pos_diffs = np.abs(np.diff(trajectory[:, :3], axis=0))
        max_step_sizes[:3] = np.ones(3) * np.max(pos_diffs)

        # For orientation (quaternion), calculate orientation error between consecutive waypoints
        ori_errors = np.zeros((trajectory.shape[0]-1, 3))
        for i in range(trajectory.shape[0]-1):
            # Use quaternions_orientation_error from transform_utils
            ori_errors[i] = T.quaternions_orientation_error(
                trajectory[i+1, 3:7],  # next quaternion
                trajectory[i, 3:7]     # current quaternion
            )

        # Get maximum orientation error for each axis
        max_step_sizes[3:] = np.ones(3) * np.max(np.abs(ori_errors))
    else:
        # For standard trajectories, calculate as before
        diffs = np.abs(np.diff(trajectory, axis=0))
        max_step_sizes = np.max(diffs, axis=0)

    # Replace zeros with small value 1e-8
    max_step_sizes = np.where(max_step_sizes < 1e-4, 1e-4, max_step_sizes)

    return max_step_sizes


def get_circular_trajectory(p1, p2, steps, revolutions=1.0, from_center=False):
    """
    Generate a circular trajectory between two points.

    Args:
        p1 (np.ndarray): Starting point [x, y, z]
        p2 (np.ndarray): Ending point [x, y, z] 
        steps (int): Number of trajectory points
        revolutions (float): Number of complete revolutions around the circle
        from_center (bool): If True, treat p1 as center and spiral outward to p2
        inverse (bool): If True, reverse the direction of rotation

    Returns:
        np.ndarray: Trajectory points of shape (steps, 3) with [x, y, z] coordinates
    """
    # Get 2D distance between points
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = np.sqrt(dx*dx + dy*dy)

    # Set up angle parameters
    if from_center:
        radius = np.linspace(0, distance, steps)
        start_angle = 0.0
    else:
        # For circular trajectory, use constant radius (half the distance)
        radius = distance / 2.0
        # Calculate center point between p1 and p2
        center_x = (p1[0] + p2[0]) / 2.0
        center_y = (p1[1] + p2[1]) / 2.0
        # Calculate start angle from center to p1
        start_angle = np.arctan2(p1[1] - center_y, p1[0] - center_x)

    # Generate angles
    angles = np.linspace(0, 2*np.pi*revolutions, steps) + start_angle

    # Calculate x, y coordinates
    if from_center:
        x = radius * np.cos(angles) + p1[0]  # Changed from p2[0] to p1[0]
        y = radius * np.sin(angles) + p1[1]  # Changed from p2[1] to p1[1]
    else:
        x = radius * np.cos(angles) + center_x
        y = radius * np.sin(angles) + center_y

    z = np.full(steps, p1[2])

    # Combine into trajectory
    trajectory = np.column_stack([x, y, z])
    return trajectory


@dataclass
class TrajectoryState:
    position: np.ndarray          # Current position
    orientation: quaternion.quaternion  # Current orientation
    linear_velocity: np.ndarray   # Linear velocity
    angular_velocity: np.ndarray  # Angular velocity
    linear_acceleration: np.ndarray    # Linear acceleration
    angular_acceleration: np.ndarray   # Angular acceleration


class MinimumJerkTrajectory:
    def __init__(self, waypoints, waypoint_rotations, duration, dt):
        """
        Initialize minimum jerk trajectory through waypoints.

        Args:
            waypoints (np.ndarray): Array of shape (N, 3) containing position waypoints
            waypoint_rotations (list): List of N quaternions for orientations
            duration (float): Total duration of trajectory
            dt (float): Time step for trajectory
        """
        self.waypoints = waypoints
        self.rotations = quaternion.as_quat_array(np.roll(waypoint_rotations, 1, axis=-1))
        self.duration = duration
        self.dt = dt
        self.num_points = len(waypoints)

        # Time parameterization
        self.times = np.linspace(0, duration, self.num_points)
        self.trajectory_times = np.arange(0, duration + dt, dt)

        # Generate trajectories
        self._generate_position_trajectory()
        self._generate_rotation_trajectory()

    def _generate_position_trajectory(self):
        """
        Generate minimum jerk trajectory for positions using cubic splines
        """
        # Create cubic spline for each dimension
        self.splines = []
        for dim in range(3):
            # Get positions for this dimension
            pos = self.waypoints[:, dim]

            # For minimum jerk, we set zero velocity and acceleration at endpoints
            # Create natural cubic spline (second derivative zero at endpoints)
            spline = CubicSpline(self.times, pos, bc_type='natural')
            self.splines.append(spline)

        # Compute positions, velocities, and accelerations
        self.positions = np.zeros((len(self.trajectory_times), 3))
        self.velocities = np.zeros_like(self.positions)
        self.accelerations = np.zeros_like(self.positions)

        for i, t in enumerate(self.trajectory_times):
            for dim in range(3):
                self.positions[i, dim] = self.splines[dim](t)
                self.velocities[i, dim] = self.splines[dim].derivative(1)(t)
                self.accelerations[i, dim] = self.splines[dim].derivative(2)(t)

    def _generate_rotation_trajectory(self):
        """
        Generate smooth quaternion trajectory using SLERP between waypoints
        """
        self.quaternions = []
        self.angular_velocities = []
        self.angular_accelerations = []

        # Time between waypoints
        segment_duration = self.duration / (self.num_points - 1)
        steps_per_segment = int(segment_duration / self.dt)

        for i in range(self.num_points - 1):
            q0 = self.rotations[i]
            q1 = self.rotations[i+1]

            # Generate interpolation parameters
            t = np.linspace(0, 1, steps_per_segment)

            # Perform SLERP for each timestep
            for ti in t:
                # SLERP
                qi = quaternion.slerp_evaluate(q0, q1, ti)
                self.quaternions.append(qi)

                # Compute angular velocity (finite difference)
                if len(self.quaternions) > 1:
                    dq = self.quaternions[-1] * self.quaternions[-2].conjugate()
                    angle = 2 * np.arccos(np.clip(dq.w, -1.0, 1.0))
                    if abs(angle) < 1e-10:
                        w = np.zeros(3)
                    else:
                        axis = np.array([dq.x, dq.y, dq.z])
                        axis = axis / np.sin(angle/2)
                        w = (angle / self.dt) * axis
                    self.angular_velocities.append(w)
                else:
                    self.angular_velocities.append(np.zeros(3))

        # Add final orientation
        self.quaternions.append(self.rotations[-1])
        self.angular_velocities.append(np.zeros(3))

        # Compute angular acceleration (finite difference)
        self.angular_accelerations = []
        for i in range(len(self.angular_velocities)):
            if i == 0:
                acc = np.zeros(3)
            else:
                acc = (self.angular_velocities[i] - self.angular_velocities[i-1]) / self.dt
            self.angular_accelerations.append(acc)

        # expose quaternion as x,y,z,w instead of the numpy-quaternion format w,x,y,z
        self.quaternions = np.roll(quaternion.as_float_array(), -1, axis=-1)

    def get_state(self, t):
        """
        Get trajectory state at time t

        Args:
            t (float): Time at which to sample trajectory

        Returns:
            TrajectoryState: State of the trajectory at time t
        """
        # Clip time to trajectory duration
        t = np.clip(t, 0, self.duration)

        # Get index for current time
        idx = int(t / self.dt)

        return TrajectoryState(
            position=self.positions[idx],
            orientation=self.quaternions[idx],
            linear_velocity=self.velocities[idx],
            angular_velocity=self.angular_velocities[idx],
            linear_acceleration=self.accelerations[idx],
            angular_acceleration=self.angular_accelerations[idx]
        )
