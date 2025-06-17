import numpy as np

#Paper:Singularity Analysis and Complete Methods to Compute the Inverse Kinematics for a 6-DOF UR/TM-Type Robot 
# https://www.mdpi.com/2218-6581/11/6/137
#UR geometric params
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

class UrCustom():
    def __init__(self, robot_type="ur5e"):
        """
        URロボットのカスタムクラス
        
        Parameters
        ----------cdd cdcdcd..
        robot_type : cd cd
            ロボットの種類
        
        Attributes
        ----------
        _a : np.array
            リンクの長さ
        _d : np.array
            リンクのオフセット
        """               

        #line 345 in ur5e_powder_grinding_default.urdf
        #  <link name="flange"/>
        # <joint name="wrist_3-flange" type="fixed">
        #     <parent link="wrist_3_link"/>
        #     <child link="flange"/>
        #     <origin rpy="0 -1.5707963267948966 -1.5707963267948966" xyz="0 0 0"/>
        # </joint>
        # <!-- ROS-Industrial 'tool0' frame: all-zeros tool frame -->
        # <link name="tool0"/>
        # <joint name="flange-tool0" type="fixed">
        #     <!-- default toolframe: X+ left, Y+ up, Z+ front -->
        #     <origin rpy="1.5707963267948966 0 1.5707963267948966" xyz="0 0 0"/>
        #     <parent link="flange"/>
        #     <child link="tool0"/>
        # </joint>
        self._l_link  = 0.1346 #[m]
        self._T_ee_tip = np.eye(4)#np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
        
        self._T_ee_tip = self._T_ee_tip @ self.trans(0, 0, 0) @ self.rot_rpy(0, -np.pi/2, -np.pi/2) #wrist3-flange
        self._T_ee_tip = self._T_ee_tip @ self.trans(0, 0, 0) @ self.rot_rpy(np.pi/2, 0, np.pi/2) #flange-tool0
        self._T_ee_tip = self._T_ee_tip @ self.trans(0, 0, self._l_link) @ self.rot_rpy(0, 0, 0) #to the tip

        self._T_eef_pose = np.array([[-1.0,0,0,0],
                                    [ 0,  1.0, 0, 0],
                                    [0, 0,-1,0],
                                    [0,0,0,1]])

        #transform from the world frame to the base frame
        #line 380 in ur5e_powder_grinding_default.urdf
        #linear translation
        #  <joint name="osx_robot_joint" type="fixed">
        # <parent link="osx_ground"/>
        # <child link="base_link"/>
        # <origin rpy="0.0 0.0 0.0" xyz="0 0 0.8"/>
        # </joint>
        #  <link name="base"/>
        # <joint name="base_link-base_fixed_joint" type="fixed">
        #     <!-- Note the rotation over Z of pi radians: as base_link is REP-103
        #         aligned (ie: has X+ forward, Y+ left and Z+ up), this is needed
        #         to correctly align 'base' with the 'Base' coordinate system of
        #         the UR controller.
        #     -->
        #     <origin rpy="0 0 3.141592653589793" xyz="0 0 0"/>
        #     <parent link="base_link"/>
        #     <child link="base"/> 0.11706499
        # </joint>
        #self._T_world2base = np.array([[-1,0,0,-0.56225077],[0,-1,0,0.0010243],[0,0,1,0.91706499],[0,0,0,1]])#np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0.8],[0,0,0,1]])
        self._T_world2base = np.eye(4)
        #self._T_world2base = self._T_world2base @ self.trans(0, 0, 0.0) @ self.rot_rpy(0, 0, 0) #world to base
        self._T_world2base = self._T_world2base @ self.trans(0, 0, 0) @ self.rot_rpy(0, 0, np.pi) #world orientation to base orientation

        self._T_compensate = np.array([[ 0,1,0,0],
                                    [ 1,0, 0,0],
                                    [0, 0, -1,0],
                                    [0,0,0,1]]
                                    )

        #to adjust the shoulder frame difference.
        self.Rx_90 = np.eye(4)
        self.Rx_90[:3, :3] = np.array([
                        [1,  0,  0],
                        [0,  0, -1],
                        [0,  1,  0]
                    ])
        
        if robot_type == "ur5e":
            self._a = np.array([0, -0.4250, -0.3922, 0, 0, 0])
            self._d = np.array([0.163, 0, 0, 0.1333, 0.0997, 0.0985])#])
            self._alpha = np.array([np.pi/2.0,0,0,np.pi/2,-np.pi/2,0])

            self._rpy = np.array([[0,0,0]])
            self._xyz = np.array([[0,0,0.163]])
            self._axis = np.array([[0,0,1] for _ in range(6)])
        elif robot_type == "ur10e":
            self._a = np.array([0, -0.6127, -0.57155, 0, 0, 0])
            self._d = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
            self._alpha = np.array([np.pi/2.0,0,0,np.pi/2,-np.pi/2,0])
        elif robot_type == "fr5":
            self._a = np.array([0, -0.425, -0.395, 0, 0, 0])
            self._d = np.array([0.152, 0, 0, 0.102, 0.102, 0.100])
            self._alpha = np.array([np.pi/2.0,0,0,np.pi/2,-np.pi/2,0])
        else:
            print("robot type is not defined")
            exit(-1)
    
    def dh(self,theta,a,d,alpha):
        """DH parameter-based transformation matrix"""
        t00=np.cos(theta)
        t01=-1.0*np.sin(theta)*np.cos(alpha)
        t02=np.sin(theta)*np.sin(alpha)
        t03=a*np.cos(theta)
        
        t10=np.sin(theta)
        t11=np.cos(theta)*np.cos(alpha)
        t12=-1.0*np.cos(theta)*np.sin(alpha)
        t13=a*np.sin(theta)

        t20=0
        t21=np.sin(alpha)
        t22=np.cos(alpha)
        t23=d

        return np.array([[t00,t01,t02,t03],[t10,t11,t12,t13],[t20,t21,t22,t23],[0,0,0,1]])
    
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

    def _mat01(self, theta1):
        self._T01 = self.dh(theta=theta1,a=self._a[0],d=self._d[0],alpha=self._alpha[0])

    def _mat12(self, theta2):
        self._T12 = self.transself.dh(theta=theta2,a=self._a[1],d=self._d[1],alpha=self._alpha[1])
        
    def _mat23(self, theta3):
        self._T23 = self.dh(theta=theta3,a=self._a[2],d=self._d[2],alpha=self._alpha[2])
        
    def _mat34(self, theta4):
        self._T34 = self.dh(theta=theta4,a=self._a[3],d=self._d[3],alpha=self._alpha[3])
        
    def _mat45(self, theta5):
        self._T45 = self.dh(theta=theta5,a=self._a[4],d=self._d[4],alpha=self._alpha[4])
        
    def _mat56(self, theta6):
        self._T56 = self.dh(theta=theta6,a=self._a[5],d=self._d[5],alpha=self._alpha[5])
        
    def mat(self, joints):
        self._mat01(joints[0])
        self._mat12(joints[1])
        self._mat23(joints[2])
        self._mat34(joints[3])
        self._mat45(joints[4])
        self._mat56(joints[5])
    

        
    def _cal_pose1(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T01 = self._T_world2base@self._T01
        n1 = self.rotMat2rotVec(rotMat=self._T01[:3,:3])
        
        self.pose1 = np.array([self._T01[0, 3], self._T01[1, 3], self._T01[2, 3], n1[0], n1[1], n1[2]])
        print(f"pose1={self.pose1}")
        
    def _cal_pose2(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T02 = self._T01 @ self._T12
        n2 = self.rotMat2rotVec(rotMat=self._T02[:3,:3])
        
        self.pose2 = np.array([self._T02[0, 3], self._T02[1, 3], self._T02[2, 3], n2[0], n2[1], n2[2]])
        print(f"pose2={self.pose2}")
        
    def _cal_pose3(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T03 = self._T01 @ self._T12 @ self._T23
        n3 = self.rotMat2rotVec(rotMat=self._T03[:3,:3])
        
        self.pose3 = np.array([self._T03[0, 3], self._T03[1, 3], self._T03[2, 3], n3[0], n3[1], n3[2]])
        print(f"pose3={self.pose3}")
        
    def _cal_pose4(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T04 = self._T01 @ self._T12 @ self._T23 @ self._T34
        n4 = self.rotMat2rotVec(rotMat=self._T04[:3,:3])
        
        self.pose4 = np.array([self._T04[0, 3], self._T04[1, 3], self._T04[2, 3], n4[0], n4[1], n4[2]])
        print(f"pose4={self.pose4}")
        
    def _cal_pose5(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T05 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45
        n5 = self.rotMat2rotVec(rotMat=self._T05[:3,:3])
        
        self.pose5 = np.array([self._T05[0, 3], self._T05[1, 3], self._T05[2, 3], n5[0], n5[1], n5[2]])
        print(f"pose5={self.pose5}")
        
    def _cal_pose6(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T06 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45 @ self._T56
        #add the tool to the end-effector.
        self._T06 = self._T06 @self._T_ee_tip #extend the end-effector to the tip of the bar.
        #self._T06 = self._T06@self._T_eef_pose
        
        #T_tip = self._T_world2base @self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45 @ self._T56 @ self._T_6_tool
        
        n6 = self.rotMat2rotVec(rotMat=self._T06[:3,:3])
        
        self.pose6 = np.array([self._T06[0, 3], self._T06[1, 3], self._T06[2, 3], n6[0], n6[1], n6[2]])

        print(f"pose6={self.pose6}")

        #n6 = self.rotMat2rotVec(rotMat=T_tip[:3,:3])
        #pose_tip = np.array([T_tip[0, 3], T_tip[1, 3], T_tip[2, 3], n6[0], n6[1], n6[2]])
        #print(f"estimated tool pose={pose_tip}")

    def cal_pose_all(self, joints):
        #self.mat(joints)

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
        p_eef = self.pose6[:3] #eef position
        r=p_eef-p_i
        J_l = np.cross(b,r) #linear translation part
        J_a=b.copy() #angular part
        J_col = np.hstack((J_l,J_a)) #(6,)
        
        return J_col
    
    def cal_jacobian01(self, joints):
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]
        
        self.J10 = self._jacobian(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, px, py, pz, 1)
        
    def cal_jacobian02(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]
        
        self.J02 = self._jacobian(s1, c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, px, py, pz, 2)
    
    def cal_jacobian03(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J03 = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 3)
        
    def cal_jacobian04(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self._T06[0, 3]
        py = self._T06[1, 3]
        pz = self._T06[2, 3]
        
        self.J04 = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 4)
            
    def cal_jacobian05(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J05 = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 5)
        
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
        print(f"Estimated end-effector pose : {self.pose6}")
        
        J1 = self.cal_jacobian_column(self._T01) #1th column
        J2 = self.cal_jacobian_column(self._T02)
        J3 = self.cal_jacobian_column(self._T03)
        J4 = self.cal_jacobian_column(self._T04)
        J5 = self.cal_jacobian_column(self._T05)
        J6 = self.cal_jacobian_column(self._T06) #6th column

        J = np.column_stack((J1, J2, J3, J4, J5, J6))  # Shape: (6,6)
        self.J = J.copy()

        # s1 = np.sin(joints[0])
        # c1 = np.cos(joints[0])
        # s23 = np.sin(joints[1] + joints[2])
        # c23 = np.cos(joints[1] + joints[2])
        # s234 = np.sin(joints[1] + joints[2] + joints[3])
        # c234 = np.cos(joints[1] + joints[2] + joints[3])
        # s5 = np.sin(joints[4])
        # c5 = np.cos(joints[4])
        # r13 = self._T06[0, 2]
        # r23 = self._T06[1, 2]
        # r33 = self._T06[2, 2]
        # px = self._T06[0, 3]
        # py = self._T06[1, 3]
        # pz = self._T06[2, 3]
        # 
        # self.J = self._jacobian(s1=s1, c1=c1, s23=s23, c23=c23, s234=s234, 
        # c234=c234, s5=s5, c5=c5, r13=r13, r23=r23, r33=r33, 
        # px=px, py=py, pz=pz, index=6)
        
    def _jacobian(self, s1, c1, s23, c23, s234, 
    c234, s5, c5, r13, r23, r33, 
    px, py, pz, index):
        j00 = -py
        j10 = px
        j20 = 0.0
        j30 = 0.0
        j40 = 0.0
        j50 = 1.0
        j01 = -c1 * (pz - self._d[0])
        j11 = -s1 * (pz - self._d[0])
        j21 = s1 * py + c1 * px
        j31 = s1
        j41 = -c1
        j51 = 0.0
        j02 = c1 * (s234 * s5 * self._d[5] + c234 * self._d[4] - s23 * self._a[2])
        j12 = s1 * (s234 * s5 * self._d[5] + c234 * self._d[4] - s23 * self._a[2])
        j22 = -c234 * s5 * self._d[5] + s234 * self._d[4] + c23 * self._a[2]
        j32 = s1
        j42 = -c1
        j52 = 0.0
        j03 = c1 * (s234 * s5 * self._d[5] + c234 * self._d[4])
        j13 = s1 * (s234 * s5 * self._d[5] + c234 * self._d[4])
        j23 = -c234 * s5 * self._d[5] + s234 * self._d[4]
        j33 = s1
        j43 = -c1
        j53 = 0.0
        j04 = -self._d[5] * (s1 * s5 + c1 * c234 * c5)
        j14 = self._d[5] * (c1 * s5 - s1 * c234 * c5)
        j24 = -c5 * s234 * self._d[5]
        j34 = c1 * s234
        j44 = s1 * s234
        j54 = -c234
        j05 = 0.0
        j15 = 0.0
        j25 = 0.0
        j35 = r13
        j45 = r23
        j55 = r33
        
        j = np.array([
            [j00, j01, j02, j03, j04, j05],
            [j10, j11, j12, j13, j14, j15],
            [j20, j21, j22, j23, j24, j25],
            [j30, j31, j32, j33, j34, j35],
            [j40, j41, j42, j43, j44, j45],
            [j50, j51, j52, j53, j54, j55]
        ])
        
        return j[:, :index]
    
    def determinant(self, j):
        """
        ヤコビアンの行列式を求める
        
        Parameters
        ----------
        j : np.array
            ヤコビアン
        
        Returns
        -------
        det : float
            行列式
        """
        
        if j.shape[0] == j.shape[1]:
            return np.linalg.det(j)

        jT = np.transpose(j)
        jTj = jT @ j
        
        return np.sqrt(np.linalg.det(jTj))

def quaternion_to_rotation_vector(q):
    """Calculate rotation vector from quaternion.

    Parameters
    --------------
    q (numpy.ndarray) : quaternion
        q.x,q.y,q.z,q.w
    
    Returns:
    -------------
    rotation_vector (numpy.ndarray) : rotation vector
    """
    # Extract the quaternion components
    x, y, z,w = q
    
    # Compute the angle theta (in radians)
    theta = 2 * np.arccos(w)
    
    # Handle the case where the quaternion represents zero rotation (w = 1)
    if np.isclose(theta, 0):
        return np.array([0, 0, 0])
    
    # Compute the axis of rotation
    axis = np.array([x, y, z])
    axis /= np.linalg.norm(axis)  # Normalize the axis
    
    # The rotation vector is theta times the axis
    rotation_vector = theta * axis
    
    return rotation_vector

def rotation_matrix_to_vector(R):
    """
    Converts a 3x3 rotation matrix to a rotation vector (axis-angle).
    
    Args:
        R (np.ndarray): 3x3 rotation matrix.
    
    Returns:
        np.ndarray: Rotation vector (3x1) — axis * angle (in radians).
    """
    # Ensure the matrix is valid
    assert R.shape == (3, 3)

    # Compute the angle
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0):
        # No rotation
        return np.zeros(3)

    elif np.isclose(theta, np.pi):
        # Special case: angle is pi
        R_plus_I = R + np.eye(3)
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
        rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
        axis = np.array([rx, ry, rz])
        return theta * axis

def rotation_vector_to_matrix(r):
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

def main():
    from ur_pykdl.ur_pykdl import ur_kinematics

    #URDF-based kinematics
    ur_kinematics_ = ur_kinematics(
        robot='ur5e_powder_grinding_default',
        rospackage='osx_powder_grinding',
        base_link='base_link',
        ee_link='gripper_tip_link',#'gripper_tip_link',#
    )

    #customized UR kinematics
    ur_kin = UrCustom()

    q = np.zeros(6)#[1.0, -1.57, 1.57, 0.0, 1.57, 1.0]  # example joint angles in radians
    q[1]=np.pi/3

    
    #URDF-based
    joint_poses = ur_kinematics_.forward(q) # all the joint pose.
    print(joint_poses)
    rotVec = quaternion_to_rotation_vector(joint_poses[3:])
    print(f"end-effector pose={joint_poses[:3]}, rotVec={rotVec}") 

    rotMat_urdf = rotation_vector_to_matrix(rotVec)

    #customized one.
    ur_kin.cal_jacobian(q)
    rotMat_dh = ur_kin._T06[:3,:3]

    rotMat_transform = np.linalg.inv(rotMat_dh)@rotMat_urdf
    print(rotMat_transform)



if __name__=="__main__":
    main()