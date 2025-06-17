import numpy as np

#Paper:Singularity Analysis and Complete Methods to Compute the Inverse Kinematics for a 6-DOF UR/TM-Type Robot 
# https://www.mdpi.com/2218-6581/11/6/137
#UR geometric params
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
class UrCustom():
    def __init__(self):
        """
        Customized UR5e kinematics. Forward kinematics and jacobian matrix.
        Robotic parameters are from /osx_powder_grinding/urdf/ur5e_powder_grinding_default.urdf
        """               

        #world frame to the robot's
        self._T_world2base = np.eye(4)
        self._T_world2base =self._T_world2base @ self.trans(-0.5622, 0.00063, 0.91667) @ self.rot_rpy(0, 0, 0)
        self._T_base20 = np.eye(4)
        self._T_base20 = self._T_base20 @ self.trans(0, 0, 0) @ self.rot_rpy(0, 0, np.pi) #world orientation to base orientation

        #robot's wrist3 to the end-effector
        self._l_link  = 0.1346 #[m]
        self._T_ee_tip = np.eye(4)#np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
        
        self._T_ee_tip = self._T_ee_tip @ self.trans(0, 0, 0) @ self.rot_rpy(0, -np.pi/2, -np.pi/2) #wrist3-flange
        self._T_ee_tip = self._T_ee_tip @ self.trans(0, 0, 0) @ self.rot_rpy(np.pi/2, 0, np.pi/2) #flange-tool0
        self._T_tool0_tip = self.trans(0, 0, self._l_link) @ self.rot_rpy(0, 0, 0) #tool0 > gripper_tip_link
        self._T_ee_tip = self._T_ee_tip @ self._T_tool0_tip #to the tip


        #For pseudo inverse jacobian.
        #This parameters are valid for UR5e.
        self._det_threshold_upper = 2.0 * 10**-2
        self._det_threshold_lower = 1.8 * 10**-2
        self._K_det = 0.1
    
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
        R = self.rot_z(gamma) @ self.rot_y(beta) @ self.rot_x(alpha)
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
    
    def rot_y(self,beta):
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
            return axis#theta * axis

        else:
            # General case
            rx = (rotMat[2, 1] - rotMat[1, 2]) / (2 * np.sin(theta))
            ry = (rotMat[0, 2] - rotMat[2, 0]) / (2 * np.sin(theta))
            rz = (rotMat[1, 0] - rotMat[0, 1]) / (2 * np.sin(theta))
            axis = np.array([rx, ry, rz])
            return axis#theta * axis
    
    def _urdf01(self,theta1):#base
        self._T01 = np.eye(4)
        self._T01 = self._T01 @ self.trans(0,0,0.163)@self.rot_rpy(0,0,0)@self.rot_z(theta1)
    
    def _urdf12(self,theta2):#shoulder
        self._T12 = np.eye(4)
        self._T12 = self._T12 @ self.trans(0,0,0)@self.rot_rpy(np.pi/2,0,0)@self.rot_z(theta2)
    
    def _urdf23(self,theta3):#elbow
        self._T23 = np.eye(4)
        self._T23 = self._T23 @ self.trans(-0.425,0,0)@self.rot_rpy(0,0,0)@self.rot_z(theta3)
    
    def _urdf34(self,theta4):#wrist1
        self._T34 = np.eye(4)
        self._T34 = self._T34 @ self.trans(-0.3922,0,0.1333)@self.rot_rpy(0,0,0)@self.rot_z(theta4)
    
    def _urdf45(self,theta5):#wrist2
        self._T45 = np.eye(4)
        self._T45 = self._T45 @ self.trans(0,-0.0997,0)@self.rot_rpy(np.pi/2,0,0)@self.rot_z(theta5)
    
    def _urdf56(self,theta6):#wrist3
        self._T56 = np.eye(4)
        self._T56 = self._T56 @ self.trans(0,0.0985,0)@self.rot_rpy(np.pi/2,np.pi,np.pi)@self.rot_z(theta6)
    
    def _mat_urdf(self, joints):#whole original robot joints.
        self._urdf01(joints[0])
        self._urdf12(joints[1])
        self._urdf23(joints[2])
        self._urdf34(joints[3])
        self._urdf45(joints[4])
        self._urdf56(joints[5])
    
    def _cal_pose1(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T01 = self._T_world2base@self._T_base20 @self._T01
        n1 = self.rotMat2rotVec(rotMat=self._T01[:3,:3])
        
        self.pose1 = np.array([self._T01[0, 3], self._T01[1, 3], self._T01[2, 3], n1[0], n1[1], n1[2]])
        #print(f"pose1={self.pose1}")
        
    def _cal_pose2(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T02 = self._T01 @ self._T12
        n2 = self.rotMat2rotVec(rotMat=self._T02[:3,:3])
        
        self.pose2 = np.array([self._T02[0, 3], self._T02[1, 3], self._T02[2, 3], n2[0], n2[1], n2[2]])
        #print(f"pose2={self.pose2}")
        
    def _cal_pose3(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T03 = self._T01 @ self._T12 @ self._T23
        n3 = self.rotMat2rotVec(rotMat=self._T03[:3,:3])
        
        self.pose3 = np.array([self._T03[0, 3], self._T03[1, 3], self._T03[2, 3], n3[0], n3[1], n3[2]])
        #print(f"pose3={self.pose3}")
        
    def _cal_pose4(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T04 = self._T01 @ self._T12 @ self._T23 @ self._T34
        n4 = self.rotMat2rotVec(rotMat=self._T04[:3,:3])
        
        self.pose4 = np.array([self._T04[0, 3], self._T04[1, 3], self._T04[2, 3], n4[0], n4[1], n4[2]])
        #print(f"pose4={self.pose4}")
        
    def _cal_pose5(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T05 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45
        n5 = self.rotMat2rotVec(rotMat=self._T05[:3,:3])
        
        self.pose5 = np.array([self._T05[0, 3], self._T05[1, 3], self._T05[2, 3], n5[0], n5[1], n5[2]])
        #print(f"pose5={self.pose5}")
        
    def _cal_pose6(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T06 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45 @ self._T56
        #add the tool to the end-effector.
        self._T06 = self._T06 @self._T_ee_tip #tool0->gripper_tip_link. extend the end-effector to the tip of the bar.
        
        n6 = self.rotMat2rotVec(rotMat=self._T06[:3,:3])
        
        self.pose6 = np.array([self._T06[0, 3], self._T06[1, 3], self._T06[2, 3], n6[0], n6[1], n6[2]])

    def cal_pose_all(self, joints):
        #URDF pattern : Universal Robots Description Format.
        self._mat_urdf(joints)
        
        self._cal_pose1(joints, False)
        self._cal_pose2(joints, False)
        self._cal_pose3(joints, False)
        self._cal_pose4(joints, False)
        self._cal_pose5(joints, False)
        self._cal_pose6(joints, False)
    
    def cal_jacobian_column(self,P):
        """Calculate the col-th column of the jacobian using b an r

        Parameters:
            col : (int)
                index of the column
            b : (numpy.ndarray)
                third column of rotational matrix, R_(col-1)
            r : 
                subtracting the translation vector i-th position from the end-effector position.
        """
        b = P[:3,2] #z-th column 
        p_i= P[:3,3] #translation element.
        p_eef = self._T06[:3,3] #eef position
        r=p_eef-p_i
        J_l = np.cross(b,r) #linear translation part
        J_a=b.copy() #angular part
        J_col = np.hstack((J_l,J_a)) #(6,)
        
        return J_col
      
    def cal_jacobian(self, joints):
        """
        ヤコビアンを求める
        
        Parameters
        ----------
        joints : np.array
            関節角度
        
        Attributes
        ----------
        J : np.array
            ヤコビアン
        """

        #calculate forward kinematics first.
        self.cal_pose_all(joints)
        #print(f"Estimated end-effector pose : {self.pose6}")
        
        J1 = self.cal_jacobian_column(self._T01) #1th column
        J2 = self.cal_jacobian_column(self._T02)
        J3 = self.cal_jacobian_column(self._T03)
        J4 = self.cal_jacobian_column(self._T04)
        J5 = self.cal_jacobian_column(self._T05)
        J6 = self.cal_jacobian_column(self._T06) #6th column

        J = np.column_stack((J1, J2, J3, J4, J5, J6))  # Shape: (6,6)
        self.J = J.copy()
    
    def determinant(self, j):
        """
        Calculate the determinant.
        
        Parameters
        ----------
        j : np.array
            Jacobian matrix
        
        Returns
        -------
        det : float
            Determinant
        """
        
        if j.shape[0] == j.shape[1]:
            return np.linalg.det(j)

        jT = np.transpose(j)
        jTj = jT @ j
        
        return np.sqrt(np.linalg.det(jTj))
    
    def _inv_jacobian(self, J):
        """
        Calculate the pseudo inverse jacobian matrix

        Parameters
        ----------
        J : np.array
            jacobian matrix
        
        Returns
        -------
        J_inv : np.array
            pseudo inverse jacobian.
        """
        
        JT = np.transpose(J)
        JTJ = JT @ J
 
        c = np.sqrt(np.linalg.det(JTJ))

        if c > self._det_threshold_upper:
            return np.linalg.inv(J)
        else:
            l = self._fanc_c(c)
            rows = np.shape(J)[0]
            
            matrix_avoid_singular = np.eye(rows) * l
            
            A = JTJ + matrix_avoid_singular
            J_inv = np.linalg.inv(A)
            return J_inv @ JT
        
    def _func_c(self, c):
        """
        For dynamic epsilon in inverse jacobian matrix.
        damped least squares method.
        
        Parameters
        ----------
        c : float
            determinant of jacobian
            
        Returns
        -------
        float
            epsilon
        """        
        
        if self._det_threshold_lower <= c and c <= self._det_threshold_upper:
            return self._K_det * np.cos(np.pi / 2.0) * (c - self._det_threshold_lower) / (self._det_threshold_upper - self._det_threshold_lower)
        elif c < self._det_threshold_lower:
            return self._K_det
        else:
            return 0.0