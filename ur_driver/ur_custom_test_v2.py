import numpy as np
from urdfpy import URDF
from ur_pykdl.ur_pykdl import ur_kinematics

class MatOpe():
    def __init__(self):
        print("Class for dealing with transformation matrix.")

    def transform_from_pose(self, xyz, rpy):
        """Create a 4x4 homogeneous transform from translation and roll-pitch-yaw."""
        T = np.eye(4)
        T[0:3, 3] = xyz
        T[0:3, 0:3] = self.rot_rpy(alpha=rpy[0],beta=rpy[1],gamma=rpy[2])
        return T

    def rot_rpy(self,alpha,beta, gamma):
        """
        Compute 4*4 rotation matrix from roll (alpha), pitch (beta), yaw (gamma).
        Angles are in radians.
        Rotation order: X (roll), Y (pitch), Z (yaw) -- R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
        Parameters:
        -----------
            alpha, beta, gamma : float
                roll, pitch, yaw
        
        Returns:
        -----------
            R : numpy.ndarray (4,4)
                Homogeneous Rotation matrix
        """
        ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
        
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0,0],
            [0, ca, -sa,0],
            [0, sa, ca,0],
            [0,0,0,1]
        ])
        Ry = np.array([
            [cb, 0, sb,0],
            [0, 1, 0,0],
            [-sb, 0, cb,0],
            [0,0,0,1]
        ])
        Rz = np.array([
            [cg, -sg, 0,0],
            [sg, cg, 0,0],
            [0, 0, 1,0],
            [0,0,0,1]
        ])
        
        # Compose rotations: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
        return R
    
    def rot_z(self,gamma):
        """ calculate the rotation matrix around z axis.

        Parameters:
        -----------
            alpha, beta, gamma : float
                roll, pitch, yaw
        
        Returns:
        -----------
            R : numpy.ndarray (4,4)
                Homogeneous Rotation matrix
        """
        cg = np.cos(gamma)
        sg = np.sin(gamma)

        Rz = np.array([
            [cg, -sg, 0,0],
            [sg, cg, 0,0],
            [0, 0, 1,0],
            [0,0,0,1]
        ])

        return Rz
    
    def rot_x(self,alpha):
        """ calculate the rotation matrix around z axis.

        Parameters:
        -----------
            alpha, beta, gamma : float
                roll, pitch, yaw
        
        Returns:
        -----------
            R : numpy.ndarray (4,4)
                Homogeneous Rotation matrix
        """
        ca= np.cos(alpha)
        sa= np.sin(alpha)

        Rx = np.array([
            [1, 0, 0,0],
            [0, ca, -sa,0],
            [0, sa, ca,0],
            [0,0,0,1]
        ])

        return Rx
    
    def rot_y(self,alpha):
        """ calculate the rotation matrix around z axis.

        Parameters:
        -----------
            alpha, beta, gamma : float
                roll, pitch, yaw
        
        Returns:
        -----------
            R : numpy.ndarray (4,4)
                Homogeneous Rotation matrix
        """
        cb = np.cos(beta)
        sb = np.sin(beta)

        Ry = np.array([
            [cb, 0, sb,0],
            [0, 1, 0,0],
            [-sb, 0, cb,0],
            [0,0,0,1]
        ])

        return Ry
    
    def trans(self,alpha,beta, gamma):
        """
        Compute 4*4 translation matrix from x (alpha), y (beta), z (gamma).
        Angles are in meters.
        
        Parameters:
        -----------
            alpha, beta, gamma : float
                x,y,z
        
        Returns:
        -----------
            R : numpy.ndarray (4,4)
                Homogeneous Rotation matrix
        """
        T = np.eye(4)
        T[0,3]=alpha
        T[1,3]=beta
        T[2,3]=gamma

        return T
    
    def rotMat2rotVec(self,rotMat):
        """
        Converts a 3x3 rotation matrix to a rotation vector (axis-angle).
        
        Args:
            R (np.ndarray): 3x3 rotation matrix.
        
        Returns:
            np.ndarray: Rotation vector (3x1) — axis * angle (in radians).
        """
        # Ensure the matrix is valid
        assert rotMat.shape == (3, 3)

        # Compute the angle
        cos_theta = (np.trace(rotMat) - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
        theta = np.arccos(cos_theta)

        if np.isclose(theta, 0):
            # No rotation
            return np.zeros(3)

        elif np.isclose(theta, np.pi):
            # Special case: angle is pi
            R_plus_I = rotMat + np.eye(3)
            axis = np.zeros(3)
            for i in range(3):
                if not np.isclose(R_plus_I[i, i], 0):
                    axis[i] = np.sqrt(R_plus_I[i, i] / 2.0)
                    break
            # Determine correct signs
            axis = axis / np.linalg.norm(axis)
            return theta * axis

        else:
            # General case
            rx = (rotMat[2, 1] - rotMat[1, 2]) / (2 * np.sin(theta))
            ry = (rotMat[0, 2] - rotMat[2, 0]) / (2 * np.sin(theta))
            rz = (rotMat[1, 0] - rotMat[0, 1]) / (2 * np.sin(theta))
            axis = np.array([rx, ry, rz])
            return theta * axis
    
    def _urdf01(self,theta1):
        self._T01 = trans(0,0,0)@rot_rpy(0,0,no.pi)@rot_z()

    def rotation_vector_to_matrix(self,r):
        """
        Convert a rotation vector to a 3x3 rotation matrix using Rodrigues' formula.

        Args:
            r (np.ndarray): Rotation vector (3,)

        Returns:
            np.ndarray: Rotation matrix (3,3)
        """
        theta = np.linalg.norm(r)
        
        if np.isclose(theta, 0):
            return np.eye(3)  # No rotation

        # Normalize the rotation axis
        k = r / theta

        # Skew-symmetric matrix of k
        K = np.array([
            [    0, -k[2],  k[1]],
            [ k[2],     0, -k[0]],
            [-k[1],  k[0],    0]
        ])

        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

def forward_kinematics_urdf(robot, joint_angles, base_link_name, ee_link_name):
    """
    Compute forward kinematics from base_link to ee_link using URDF parameters.

    Parameters:
    - robot: urdfpy.URDF object
    - joint_angles: dict {joint_name: angle_in_radians}
    - base_link_name: str, name of the base link
    - ee_link_name: str, name of the end-effector link

    Returns:
    - 4x4 numpy array representing the pose of ee_link in base_link frame
    """

    _matop = MatOpe()

    # Find the chain of links and joints from base_link to ee_link
    chain = robot.get_chain(base_link_name, ee_link_name, links=False, fixed=False)

    T = np.eye(4)  # Initialize transform as identity

    for joint_name in chain:
        joint = robot.joint_map[joint_name]
        print(joint)

        # Fixed transform from parent link to joint frame (origin)
        T_joint_origin = _matop.transform_from_pose(joint.origin.xyz, joint.origin.rpy)

        # Rotation about joint axis by joint angle (if revolute or continuous)
        if joint.joint_type in ['revolute', 'continuous']:
            q = joint_angles.get(joint_name, 0.0)
            axis = joint.axis
            R_joint = R.from_rotvec(q * np.array(axis)).as_matrix()
        else:
            # For fixed joints or prismatic (not handled here), no rotation
            R_joint = np.eye(3)

        T_joint_motion = np.eye(4)
        T_joint_motion[0:3, 0:3] = R_joint

        # Total transform for this joint
        T = T @ T_joint_origin @ T_joint_motion

    return T

# Example usage:
if __name__ == "__main__":
    matop_ =MatOpe()

    # Load your URDF file
    robot = URDF.load("/root/osx-ur/catkin_ws/src/osx_powder_grinding/osx_powder_grinding/urdf/ur5e_powder_grinding_default.urdf")

    for joint in robot.joints:
       print('{} connects {} to {}'.format(
       joint.name, joint.parent, joint.child
        ))

    q = np.zeros(6)#[1.0, -1.57, 1.57, 0.0, 1.57, 1.0]  # example joint angles in radians
    q[1]=np.pi/3
    #     # Define joint angles in radians for all joints in the chain
    #     joint_angles = {
    #         'shoulder_pan_joint': q[0],
    #         'shoulder_lift_joint': q[1],
    #         'elbow_joint': q[2],
    #         'wrist_1_joint': q[3],
    #         'wrist_2_joint': q[4],
    #         'wrist_3_joint': q[5],
    #     }
    # 
    #     base_link = "base_link"
    #     ee_link = "gripper_tip_link"
    # 
    #     pose = forward_kinematics_urdf(robot, joint_angles, base_link, ee_link)
    # 
    #     print("End-effector pose (4x4 matrix):\n", pose)


    #URDF-based kinematics
    ur_kinematics_ = ur_kinematics(
        robot='ur5e_powder_grinding_default',
        rospackage='osx_powder_grinding',
        base_link='base_link',
        ee_link='gripper_tip_link',#'gripper_tip_link',#
    )

    
    #URDF-based
    joint_poses = ur_kinematics_.forward(q) # all the joint pose.
    rotVec = quaternion_to_rotation_vector(joint_poses[3:])
    print(f"end-effector pose={joint_poses[:3]}, rotVec={rotVec}") 
    R_act = matop_.rotation_vector_to_matrix(rotVec)
    T_act = np.eye(4)
    T_act[:3,:3] = R_act
    T_act[:3,3] = joint_poses[:3]
    print(f"Actual pose = {T_act}")