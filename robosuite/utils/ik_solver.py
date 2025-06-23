import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.optimize import least_squares
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IKResult:
    """Data class to store IK solution results"""
    success: bool
    joint_angles: Optional[np.ndarray] = None
    error_pos: Optional[float] = None
    error_rot: Optional[float] = None
    message: str = ""
    elapsed_time: float = 0.0


class IKError(Exception):
    """Custom exception for IK-related errors"""
    pass


class TimeoutError(IKError):
    """Exception raised when IK solver times out"""
    pass


class MuJoCoIKSolver:
    def __init__(self, model: MjModel, data: MjData, end_effector_site: str,
                 position_threshold: float = 1e-4,
                 rotation_threshold: float = 1e-3,
                 time_limit: float = 1.0,
                 joint_indexes=None,
                 base_body_name: str = None):
        """
        Initialize the IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            end_effector_site: Name of the site marking the end effector
            position_threshold: Maximum acceptable position error (meters)
            rotation_threshold: Maximum acceptable rotation error (radians)
            time_limit: Maximum time allowed for IK solving (seconds)
            joint_indexes: Indexes of joints to control (default: all)
            base_body_name: Name of the robot base body for frame transformations (optional)
        """
        self.model = model
        self.data = data
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.time_limit = time_limit
        self.joint_indexes = joint_indexes if joint_indexes else [i for i in range(model.nv)]

        # For time tracking during optimization
        self._start_time = None
        self._timeout_occurred = False

        try:
            self.ee_site_id = model.site(end_effector_site).id
        except Exception as e:
            raise IKError(f"End effector site '{end_effector_site}' not found in model: {str(e)}")

        # Find base body for frame transformations
        self.base_body_id = None
        if base_body_name:
            try:
                self.base_body_id = model.body(base_body_name).id
            except Exception as e:
                logger.warning(f"Base body '{base_body_name}' not found: {str(e)}. Using world frame.")

        if self.base_body_id is None:
            # Try to find the first non-world body as base
            for i in range(1, model.nbody):  # Skip world body (index 0)
                if model.body_parentid[i] == 0:  # Direct child of world
                    self.base_body_id = i
                    logger.info(f"Using body '{model.body(i).name}' as robot base")
                    break

    def get_base_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current transform from world to base frame.

        Returns:
            Tuple of (base_position, base_rotation_matrix)
        """
        if self.base_body_id is None:
            # No base body defined, return identity transform
            return np.zeros(3), np.eye(3)

        base_pos = self.data.xpos[self.base_body_id].copy()
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3).copy()

        return base_pos, base_rot

    def transform_to_world_frame(self, pos: np.ndarray, rot: Optional[np.ndarray] = None,
                                 from_frame: str = 'base') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform pose from specified frame to world frame.

        Args:
            pos: Position vector
            rot: Rotation matrix (optional)
            from_frame: Source frame ('base' or 'world')

        Returns:
            Tuple of (world_position, world_rotation)
        """
        if from_frame == 'world':
            return pos.copy(), rot.copy() if rot is not None else None

        elif from_frame == 'base':
            base_pos, base_rot = self.get_base_transform()

            # Transform position: world_pos = base_pos + base_rot @ local_pos
            world_pos = base_pos + base_rot @ pos

            # Transform rotation: world_rot = base_rot @ local_rot
            world_rot = None
            if rot is not None:
                world_rot = base_rot @ rot

            return world_pos, world_rot

        else:
            raise ValueError(f"Unknown frame: {from_frame}. Use 'world' or 'base'")

    def transform_to_base_frame(self, pos: np.ndarray, rot: Optional[np.ndarray] = None,
                                from_frame: str = 'world') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform pose from specified frame to base frame.

        Args:
            pos: Position vector
            rot: Rotation matrix (optional)
            from_frame: Source frame ('base' or 'world')

        Returns:
            Tuple of (base_position, base_rotation)
        """
        if from_frame == 'base':
            return pos.copy(), rot.copy() if rot is not None else None

        elif from_frame == 'world':
            base_pos, base_rot = self.get_base_transform()

            # Transform position: local_pos = base_rot.T @ (world_pos - base_pos)
            local_pos = base_rot.T @ (pos - base_pos)

            # Transform rotation: local_rot = base_rot.T @ world_rot
            local_rot = None
            if rot is not None:
                local_rot = base_rot.T @ rot

            return local_pos, local_rot

        else:
            raise ValueError(f"Unknown frame: {from_frame}. Use 'world' or 'base'")

    def check_target_reachability(self, target_pos: np.ndarray) -> bool:
        """
        Perform a basic reachability check for the target position.

        Args:
            target_pos: Target position to check

        Returns:
            bool: True if target might be reachable, False if definitely unreachable
        """
        # Calculate maximum reach of the arm (sum of all link lengths)
        total_reach = 0
        for i in range(self.model.nbody):
            pos = self.model.body_pos[i]
            total_reach += np.linalg.norm(pos)

        # Check if target is within maximum reach
        base_pos = self.model.body_pos[0]  # assume first body is base
        distance_to_target = np.linalg.norm(target_pos - base_pos)

        return distance_to_target <= total_reach

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint angles.

        Args:
            q: Joint angles

        Returns:
            Tuple of end effector position and orientation

        Raises:
            IKError: If forward kinematics computation fails
        """
        try:
            # Verify joint angles are valid
            if not np.all(np.isfinite(q)):
                raise IKError("Joint angles contain NaN or inf values")

            # Set joint positions
            self.data.qpos[self.joint_indexes] = q

            # Compute forward kinematics
            mujoco.mj_forward(self.model._model, self.data._data)

            # Get end effector position and orientation
            pos = self.data.site_xpos[self.ee_site_id].copy()
            rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

            return pos, rot

        except Exception as e:
            raise IKError(f"Forward kinematics computation failed: {str(e)}")

    def cost_function_with_timeout(self, q: np.ndarray, target_pos: np.ndarray,
                                   target_rot: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cost function for optimization with timeout checking.

        Args:
            q: Joint angles
            target_pos: Target position
            target_rot: Target rotation matrix (optional)

        Returns:
            Error vector

        Raises:
            TimeoutError: If time limit is exceeded
        """
        # Check timeout
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed > self.time_limit:
                self._timeout_occurred = True
                raise TimeoutError(f"IK solver timed out after {elapsed:.3f} seconds")

        return self.cost_function(q, target_pos, target_rot)

    def cost_function(self, q: np.ndarray, target_pos: np.ndarray,
                      target_rot: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cost function for optimization with error handling.

        Args:
            q: Joint angles
            target_pos: Target position
            target_rot: Target rotation matrix (optional)

        Returns:
            Error vector
        """
        try:
            current_pos, current_rot = self.forward_kinematics(q)

            # Position error
            pos_error = current_pos - target_pos

            if target_rot is None:
                return pos_error

            # Rotation error using matrix logarithm
            R_error = np.dot(current_rot.T, target_rot)
            rot_error = self.matrix_log(R_error)

            return np.concatenate([pos_error, rot_error.flatten()])

        except Exception as e:
            # Return a large error vector if computation fails
            size = 3 if target_rot is None else 6
            return np.ones(size) * 1e6

    @staticmethod
    def matrix_log(R: np.ndarray) -> np.ndarray:
        """
        Matrix logarithm for rotation matrices with error handling.

        Args:
            R: Rotation matrix

        Returns:
            Vector representation of rotation
        """
        try:
            theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            if theta < 1e-10:
                return np.zeros(3)
            factor = theta / (2 * np.sin(theta))
            return factor * np.array([R[2, 1] - R[1, 2],
                                      R[0, 2] - R[2, 0],
                                      R[1, 0] - R[0, 1]])
        except Exception as e:
            raise IKError(f"Matrix logarithm computation failed: {str(e)}")

    def validate_solution(self, q: np.ndarray, target_pos: np.ndarray,
                          target_rot: Optional[np.ndarray] = None) -> Tuple[bool, float, Optional[float]]:
        """
        Validate IK solution against thresholds.

        Args:
            q: Joint angles solution
            target_pos: Target position
            target_rot: Target rotation matrix (optional)

        Returns:
            Tuple of (success, position_error, rotation_error)
        """
        current_pos, current_rot = self.forward_kinematics(q)

        pos_error = np.linalg.norm(current_pos - target_pos)
        rot_error = None

        if target_rot is not None:
            R_error = np.dot(current_rot.T, target_rot)
            rot_error = np.linalg.norm(self.matrix_log(R_error))

        success = (pos_error <= self.position_threshold and
                   (target_rot is None or rot_error <= self.rotation_threshold))

        return success, pos_error, rot_error

    def solve_ik(self, target_pos: np.ndarray, target_rot: Optional[np.ndarray] = None,
                 initial_guess: Optional[np.ndarray] = None,
                 frame: str = 'world') -> IKResult:
        """
        Solve inverse kinematics with comprehensive error handling and frame support.
        Uses multiple strategies within the time limit to find the best solution.

        Args:
            target_pos: Target position
            target_rot: Target rotation matrix (optional)
            initial_guess: Initial joint angles (optional)
            frame: Target frame ('world' or 'base')

        Returns:
            IKResult object containing solution status and details
        """
        start_time = time.time()
        self._start_time = start_time
        self._timeout_occurred = False

        try:
            # Input validation
            if not np.all(np.isfinite(target_pos)):
                raise IKError("Target position contains NaN or inf values")
            if target_rot is not None and not np.all(np.isfinite(target_rot)):
                raise IKError("Target rotation contains NaN or inf values")

            if frame not in ['world', 'base']:
                raise IKError(f"Invalid frame '{frame}'. Use 'world' or 'base'")

            # Transform target to world frame for internal computation
            if frame == 'base':
                target_pos_world, target_rot_world = self.transform_to_world_frame(
                    target_pos, target_rot, from_frame='base'
                )
            else:
                target_pos_world, target_rot_world = target_pos.copy(), target_rot.copy() if target_rot is not None else None

            # Check basic reachability
            if not self.check_target_reachability(target_pos_world):
                return IKResult(
                    success=False,
                    message="Target position appears to be outside robot's reachable workspace",
                    elapsed_time=time.time() - start_time
                )

            # Set bounds for joint angles
            bounds = (self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

            # Keep track of best solution found so far
            best_result = None
            best_error = float('inf')
            attempt_count = 0

            # Strategy 1: Try with provided/current joint angles
            initial_guesses = []
            if initial_guess is not None:
                initial_guesses.append(initial_guess.copy())
            else:
                initial_guesses.append(self.data.qpos.copy())

            # Strategy 2: Add some common robot configurations
            if len(initial_guesses) < 5:  # Add more initial guesses if we have time
                # Zero configuration
                initial_guesses.append(np.zeros(len(self.joint_indexes)))
                # Random configurations within bounds
                for _ in range(3):
                    random_config = np.random.uniform(bounds[0], bounds[1])
                    initial_guesses.append(random_config)

            # Try different optimization approaches within time limit
            methods = ['trf', 'lm', 'dogbox']
            tolerances = [(1e-8, 1e-8), (1e-6, 1e-6), (1e-4, 1e-4)]

            for method in methods:
                if time.time() - start_time >= self.time_limit * 0.9:  # Leave 10% buffer
                    break

                for ftol, xtol in tolerances:
                    if time.time() - start_time >= self.time_limit * 0.9:
                        break

                    for initial_config in initial_guesses:
                        if time.time() - start_time >= self.time_limit * 0.9:
                            break

                        attempt_count += 1
                        try:
                            # Handle warnings as errors during optimization
                            with warnings.catch_warnings(record=True):
                                warnings.simplefilter("ignore")  # Ignore warnings during multiple attempts

                                # Solve optimization problem with timeout
                                result = least_squares(
                                    fun=self.cost_function_with_timeout,
                                    x0=initial_config,
                                    args=(target_pos_world, target_rot_world),
                                    bounds=bounds,
                                    method=method,
                                    ftol=ftol,
                                    xtol=xtol,
                                    max_nfev=500  # Smaller per-attempt limit to allow multiple attempts
                                )

                                # Validate solution
                                success, pos_error, rot_error = self.validate_solution(
                                    result.x, target_pos_world, target_rot_world
                                )

                                # Calculate combined error for comparison
                                combined_error = pos_error + (rot_error if rot_error is not None else 0)

                                # If this is a valid solution, return immediately
                                if success:
                                    elapsed_time = time.time() - start_time
                                    return IKResult(
                                        success=True,
                                        joint_angles=result.x,
                                        error_pos=pos_error,
                                        error_rot=rot_error,
                                        message=f"Successfully found IK solution in {frame} frame (attempt {attempt_count})",
                                        elapsed_time=elapsed_time
                                    )

                                # Keep track of best solution so far
                                if combined_error < best_error:
                                    best_error = combined_error
                                    best_result = IKResult(
                                        success=False,
                                        joint_angles=result.x,
                                        error_pos=pos_error,
                                        error_rot=rot_error,
                                        message=f"Best solution found but exceeds error thresholds (attempt {attempt_count})",
                                        elapsed_time=time.time() - start_time
                                    )

                        except TimeoutError as e:
                            # Time limit reached
                            break
                        except Exception as e:
                            # This attempt failed, continue with next
                            logger.debug(f"Attempt {attempt_count} failed: {str(e)}")
                            continue

            # If we get here, no valid solution was found within time limit
            elapsed_time = time.time() - start_time

            if best_result is not None:
                best_result.elapsed_time = elapsed_time
                best_result.message += f" (tried {attempt_count} attempts in {elapsed_time:.3f}s)"
                return best_result
            else:
                return IKResult(
                    success=False,
                    message=f"No solution found after {attempt_count} attempts in {elapsed_time:.3f}s",
                    elapsed_time=elapsed_time
                )

        except IKError as e:
            return IKResult(
                success=False,
                message=f"IK Error: {str(e)}",
                elapsed_time=time.time() - start_time
            )
        except Exception as e:
            return IKResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                elapsed_time=time.time() - start_time
            )
        finally:
            self._start_time = None
            self._timeout_occurred = False

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Get the Jacobian matrix at the current configuration with error handling.

        Args:
            q: Joint angles

        Returns:
            Geometric Jacobian matrix

        Raises:
            IKError: If Jacobian computation fails
        """
        try:
            if not np.all(np.isfinite(q)):
                raise IKError("Joint angles contain NaN or inf values")

            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)

            # Get position Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

            return np.vstack([jacp, jacr])

        except Exception as e:
            raise IKError(f"Jacobian computation failed: {str(e)}")
