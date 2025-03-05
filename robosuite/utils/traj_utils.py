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
    # Step 1: Calculate bowl parameters
    radius = mortar_diameter / 2

    # Step 2: Verify if desired height is valid
    if desired_height > radius:
        raise ValueError("Desired height cannot be greater than bowl radius")

    # Step 3: Calculate radius of the circle at desired height
    # For upward facing bowl: r^2 = R^2 - (R-h)^2
    circle_radius = np.sqrt(radius**2 - (radius - desired_height)**2)

    # Step 4: Generate points along a circle at desired height
    theta = np.linspace(0, 2*np.pi, n_steps)

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

    # Add initial pose to the end of the trajectory to complete the circle
    trajectory = np.concatenate([trajectory, [trajectory[0]]])

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
