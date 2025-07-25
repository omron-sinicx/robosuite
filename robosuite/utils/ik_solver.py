import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.optimize import least_squares
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging

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


class IKError(Exception):
    """Custom exception for IK-related errors"""
    pass


class MuJoCoIKSolver:
    def __init__(self, model: MjModel, data: MjData, end_effector_site: str,
                 position_threshold: float = 1e-3,
                 rotation_threshold: float = 1e-2,
                 max_iterations: int = 100,
                 joint_indexes=None):
        """
        Initialize the IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            end_effector_site: Name of the site marking the end effector
            position_threshold: Maximum acceptable position error (meters)
            rotation_threshold: Maximum acceptable rotation error (radians)
            max_iterations: Maximum number of optimization iterations
        """
        self.model = model
        self.data = data
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_iterations = max_iterations
        self.joint_indexes = joint_indexes if joint_indexes else [i for i in range(model.nv)]

        try:
            self.ee_site_id = model.site(end_effector_site).id
        except Exception as e:
            raise IKError(f"End effector site '{end_effector_site}' not found in model: {str(e)}")

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
                 initial_guess: Optional[np.ndarray] = None) -> IKResult:
        """
        Solve inverse kinematics with comprehensive error handling.

        Args:
            target_pos: Target position
            target_rot: Target rotation matrix (optional)
            initial_guess: Initial joint angles (optional)

        Returns:
            IKResult object containing solution status and details
        """
        try:
            # Input validation
            if not np.all(np.isfinite(target_pos)):
                raise IKError("Target position contains NaN or inf values")
            if target_rot is not None and not np.all(np.isfinite(target_rot)):
                raise IKError("Target rotation contains NaN or inf values")

            # Check basic reachability
            if not self.check_target_reachability(target_pos):
                return IKResult(
                    success=False,
                    message="Target position appears to be outside robot's reachable workspace"
                )

            # Set initial guess
            if initial_guess is None:
                initial_guess = self.data.qpos.copy()

            # Set bounds for joint angles
            bounds = (self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

            # Handle warnings as errors during optimization
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("error")

                try:
                    # Solve optimization problem
                    result = least_squares(
                        fun=self.cost_function,
                        x0=initial_guess,
                        args=(target_pos, target_rot),
                        bounds=bounds,
                        method='trf',
                        ftol=1e-8,
                        xtol=1e-8,
                        max_nfev=self.max_iterations
                    )
                except Warning as warn:
                    logger.warning(f"Optimization warning: {str(warn)}")
                except Exception as e:
                    raise IKError(f"Optimization failed: {str(e)}")

            # Validate solution
            success, pos_error, rot_error = self.validate_solution(
                result.x, target_pos, target_rot
            )

            if not success:
                return IKResult(
                    success=False,
                    joint_angles=result.x,
                    error_pos=pos_error,
                    error_rot=rot_error,
                    message="Solution found but exceeds error thresholds"
                )

            return IKResult(
                success=True,
                joint_angles=result.x,
                error_pos=pos_error,
                error_rot=rot_error,
                message="Successfully found IK solution"
            )

        except IKError as e:
            return IKResult(
                success=False,
                message=f"IK Error: {str(e)}"
            )
        except Exception as e:
            return IKResult(
                success=False,
                message=f"Unexpected error: {str(e)}"
            )

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
