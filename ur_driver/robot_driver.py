from .ur_custom import UrCustom
import numpy as np

class RobotDriver:
    # def __init__(self, Node, robot_type, omega_max):
    def __init__(self, omega_max=6.28, robot_type="ur5e"):
        """
        順運動学や逆運動学を行うクラス
        （動くことは確認しているが何をしているのかは不明）
        
        Attributes
        ----------
        _ur_custom : UrCustom
            URロボットのカスタムクラス
        _det_threshold_upper : float
            行列式の閾値
        _det_threshold_lower : float
            行列式の閾値
        _K_det : float
            行列式の閾値
        """        
        
        # self._Node = Node
        
        self._omega_max = omega_max
        
        self._ur_custom = UrCustom(robot_type)
        
        if robot_type == "ur5e":        
            self._det_threshold_upper = 2.0 * 10**-2
            self._det_threshold_lower = 1.8 * 10**-2
        elif robot_type == "ur10e":
            self._det_threshold_upper = 2.5 * 10**-2
            self._det_threshold_lower = 2.25 * 10**-2
        elif robot_type == "fr5":
            self._det_threshold_upper = 2.0 * 10**-2
            self._det_threshold_lower = 1.8 * 10**-2   
        else:
            raise ValueError("robot_type must be 'ur5e' or 'ur10e'")
        
        self._K_det = 0.1
        
    def speed2omega(self, joint_angles, velocity):
        """
        TCP速度から関節速度を計算する
        
        Parameters
        ----------
        joint_angles : np.array
            関節角度
        velocity : np.array
            TCP速度
        
        Returns
        -------
        vel_joints : np.array
            関節速度
        
        Attributes
        ----------
        det : float
            行列式
        """
        
        self._ur_custom.cal_pose_all(joint_angles)

        self.config_ = np.array([self._ur_custom.pose1, self._ur_custom.pose2, self._ur_custom.pose3, self._ur_custom.pose4, self._ur_custom.pose5, self._ur_custom.pose6])
        
        self._ur_custom.cal_jacobian(joint_angles)
        J = self._ur_custom.J
        
        self.det = self._ur_custom.determinant(J)
        
        J_inv = self._inv_jacobian(J)
        
        vel_cv = np.transpose(velocity)
        vel = J_inv @ vel_cv
        
        vel_joints = np.zeros(np.shape(vel))
        
        for i in range(np.shape(vel_joints)[0]):
            vel_joints[i] = max(min(vel[i], self._omega_max), -self._omega_max)
            
        return vel_joints
    
    def get_tcp_pose(self, joint_angles):
        """
        関節角度からTCP座標を計算する
        
        Parameters
        ----------
        joint_angles : np.array
            関節角度

        Returns
        -------
        self._ur_custom.pose6 : np.array
            TCP座標
        """
        
        self._ur_custom.cal_pose_all(joint_angles)
        return self._ur_custom.pose6
        
    def _inv_jacobian(self, J):
        """
        ヤコビアンの逆行列を計算する
        
        Parameters
        ----------
        J : np.array
            ヤコビアン
        
        Returns
        -------
        J_inv : np.array
            ヤコビアンの逆行列
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
        
    def _fanc_c(self, c):
        """
        関数cを計算する
        
        Parameters
        ----------
        c : float
            行列式
            
        Returns
        -------
        float
            関数c
        """        
        
        if self._det_threshold_lower <= c and c <= self._det_threshold_upper:
            return self._K_det * np.cos(np.pi / 2.0) * (c - self._det_threshold_lower) / (self._det_threshold_upper - self._det_threshold_lower)
        elif c < self._det_threshold_lower:
            return self._K_det
        else:
            return 0.0