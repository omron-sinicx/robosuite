import casadi as cs

_PI = 3.141592653589793
#For pseudo inverse jacobian. This parameters are adjusted according to the robot.
#This parameters are for UR5e.
_det_threshold_upper = 2.0 * 10**-2
_det_threshold_lower = 1.8 * 10**-2
_K_det = 0.1

"""Rotation matrix operation"""
def rot_x(alpha):
    ca = cs.cos(alpha)
    sa = cs.sin(alpha)
    return cs.vertcat(
        cs.horzcat(1, 0, 0, 0),
        cs.horzcat(0, ca, -sa, 0),
        cs.horzcat(0, sa, ca, 0),
        cs.horzcat(0, 0, 0, 1)
    )

def rot_y(beta):
    cb = cs.cos(beta)
    sb = cs.sin(beta)
    return cs.vertcat(
        cs.horzcat(cb, 0, sb, 0),
        cs.horzcat(0, 1, 0, 0),
        cs.horzcat(-sb, 0, cb, 0),
        cs.horzcat(0, 0, 0, 1)
    )

def rot_z(gamma):
    cg = cs.cos(gamma)
    sg = cs.sin(gamma)
    return cs.vertcat(
        cs.horzcat(cg, -sg, 0, 0),
        cs.horzcat(sg, cg, 0, 0),
        cs.horzcat(0, 0, 1, 0),
        cs.horzcat(0, 0, 0, 1)
    )

def rot_rpy(alpha, beta, gamma):
    return cs.mtimes(cs.mtimes(rot_z(gamma), rot_y(beta)), rot_x(alpha))

"""Translation matrix operation"""
def trans(x, y, z):
    T = cs.MX.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

"""UR5e forward kinematics"""
def ur5e_fk_all_frames():
    q = cs.MX.sym("q", 6)

    #world frame to the robot's base
    T_world2base = cs.MX.eye(4)
    T_world2base = cs.mtimes(cs.mtimes(T_world2base,trans(-0.5622, 0.00063, 0.91667)),rot_rpy(0, 0, 0))

    # Base to robot base frame zero (constant)
    T_base20 = cs.MX.eye(4)
    T_base20 = cs.mtimes(cs.mtimes(T_base20,trans(0, 0, 0)),rot_rpy(0, 0, _PI))

    #6th joint to the end-effector.
    l_link = 0.1346
    T_ee_tip = cs.MX.eye(4)
    T_ee_tip = cs.mtimes(cs.mtimes(T_ee_tip, trans(0, 0, 0)), rot_rpy(0, -_PI/2, -_PI/2))  # wrist3 -> flange
    T_ee_tip = cs.mtimes(cs.mtimes(T_ee_tip, trans(0, 0, 0)),rot_rpy(_PI/2, 0, _PI/2))    # flange -> tool0
    T_ee_tip = cs.mtimes(cs.mtimes(T_ee_tip, trans(0, 0, l_link)),rot_rpy(0, 0, 0))          # tool0 -> tip


    # URDF kinematic chain
    T01 = cs.mtimes(cs.mtimes(trans(0, 0, 0.163),rot_rpy(0, 0, 0)),rot_z(q[0]))
    T12 = cs.mtimes(cs.mtimes(trans(0, 0, 0),rot_rpy(_PI/2, 0, 0)),rot_z(q[1]))
    T23 = cs.mtimes(cs.mtimes(trans(-0.425, 0, 0),rot_rpy(0, 0, 0)),rot_z(q[2]))
    T34 = cs.mtimes(cs.mtimes(trans(-0.3922, 0, 0.1333),rot_rpy(0, 0, 0)),rot_z(q[3]))
    T45 = cs.mtimes(cs.mtimes(trans(0, -0.0997, 0),rot_rpy(_PI/2, 0, 0)),rot_z(q[4]))
    T56 = cs.mtimes(cs.mtimes(trans(0, 0.0985, 0),rot_rpy(_PI/2, _PI, _PI)),rot_z(q[5]))


    # Frame transforms
    T01 = cs.mtimes(cs.mtimes(T_world2base, T_base20), T01)
    T02 = cs.mtimes(T01, T12)
    T03 = cs.mtimes(T02, T23)
    T04 = cs.mtimes(T03, T34)
    T05 = cs.mtimes(T04, T45)
    T06 = cs.mtimes(T05, T56)
    T06 = cs.mtimes(T06, T_ee_tip)

    return cs.Function("fk_all", [q], [T01, T02, T03, T04, T05, T06])

"""Calculate Jacobian matrix"""
def cal_jacobian_casadi():
    q = cs.MX.sym("q", 6)
    fk_all = ur5e_fk_all_frames() #create an instance of casadi format function.
    T01, T02, T03, T04, T05, T06 = fk_all(q)

    Ts = [T01, T02, T03, T04, T05, T06]
    J_cols = []

    p_eef = T06[0:3, 3]

    for i in range(6):
        T = Ts[i]
        z_axis = T[0:3, 2]
        p_i = T[0:3, 3]
        r = p_eef - p_i
        J_l = cs.cross(z_axis, r)
        J_a = z_axis
        J_col = cs.vertcat(J_l, J_a)
        J_cols.append(J_col)

    J = cs.horzcat(*J_cols)  # Shape (6,6)

    return cs.Function("jacobian", [q], [J])

"""Calculate the determinant"""
def jacobian_determinant_casadi():
    q = cs.MX.sym("q", 6)
    J_func = cal_jacobian_casadi()
    J = J_func(q)
    J_T = J.T
    det = cs.sqrt(cs.det(cs.mtimes(J_T, J)))
    return cs.Function("jacobian_det", [q], [det])

"""Calculate pseudo inverse jacobian."""
"""For calculating determinant."""
def minor_matrix(M, i, j):
    """Return the minor of matrix M excluding row i and column j."""
    rows = [r for r in range(M.size1()) if r != i]
    cols = [c for c in range(M.size2()) if c != j]
    return cs.blockcat([[M[r, c] for c in cols] for r in rows])

def det_mx(M):
    """Compute the determinant of an MX matrix recursively (Laplace expansion)."""
    n = M.size1()
    assert n == M.size2(), "Matrix must be square"

    if n == 1:
        return M[0, 0]
    elif n == 2:
        return M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
    else:
        det_val = 0
        for j in range(n):
            sign = (-1)**j
            M_minor = minor_matrix(M, 0, j)
            det_val += sign * M[0, j] * det_mx(M_minor)
        return det_val

def func_c_casadi(c):
    cond1 = cs.logic_and(c >= _det_threshold_lower, c <= _det_threshold_upper)
    cond2 = c < _det_threshold_lower

    # piecewise expression
    expr1 = _K_det * cs.cos(_PI / 2) * (c - _det_threshold_lower) / (_det_threshold_upper - _det_threshold_lower)
    expr2 = _K_det
    expr3 = 0.0

    return cs.if_else(cond1, expr1, cs.if_else(cond2, expr2, expr3))

def inv_jacobian_casadi():
    J = cs.MX.sym("J", 6, 6)

    JT = J.T
    JTJ = cs.mtimes(JT, J)
    #calculate determinant.
    c = cs.sqrt(det_mx(JTJ))
    #adjust eplsilon according to determinant.
    epsilon = func_c_casadi(c)
    #Leuben-marquardt method
    rows = J.size1()
    I = cs.MX.eye(rows)
    matrix_avoid_singular = epsilon * I

    A = JTJ + matrix_avoid_singular
    A_inv = cs.inv(A)
    J_inv = cs.if_else(c > _det_threshold_upper, cs.inv(J), cs.mtimes(A_inv, JT))

    return cs.Function("inv_jacobian", [J], [J_inv,epsilon])