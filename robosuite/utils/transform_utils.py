"""
Utility functions of matrix and vector transformations.

NOTE: convention for quaternions is (x, y, z, w)
"""

import math

import numpy as np

from robosuite.utils.numba import jit_decorator

PI = np.pi
EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q (np.array): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).

    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def quat_conjugate(quaternion):
    """
    Return conjugate of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion conjugate
    """
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )


def quat_inverse(quaternion):
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    E.g.:
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True

    >>> q = quat_slerp(q0, q1, 1.0)
    >>> np.allclose(q, q1)
    True

    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True

    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def random_quat(rand=None):
    """
    Return uniform random unit quaternion.

    E.g.:
    >>> q = random_quat()
    >>> np.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quat(np.random.random(3))
    >>> q.shape
    (4,)

    Args:
        rand (3-array or None): If specified, must be three independent random variables that are uniformly distributed
            between 0 and 1.

    Returns:
        np.array: (x,y,z,w) random quaternion
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
        dtype=np.float32,
    )


def random_axis_angle(angle_limit=None, random_state=None):
    """
    Samples an axis-angle rotation by first sampling a random axis
    and then sampling an angle. If @angle_limit is provided, the size
    of the rotation angle is constrained.

    If @random_state is provided (instance of np.random.RandomState), it
    will be used to generate random numbers.

    Args:
        angle_limit (None or float): If set, determines magnitude limit of angles to generate
        random_state (None or RandomState): RNG to use if specified

    Raises:
        AssertionError: [Invalid RNG]
    """
    if angle_limit is None:
        angle_limit = 2.0 * np.pi

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState)
        npr = random_state
    else:
        npr = np.random

    # sample random axis using a normalized sample from spherical Gaussian.
    # see (http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)
    # for why it works.
    random_axis = npr.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    random_angle = npr.uniform(low=0.0, high=angle_limit)
    return random_axis, random_angle


def vec(values):
    """
    Converts value tuple into a numpy vector.

    Args:
        values (n-array): a tuple of numbers

    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)


def mat4(array):
    """
    Converts an array to 4x4 matrix.

    Args:
        array (n-array): the array in form of vec, list, or tuple

    Returns:
        np.array: a 4x4 numpy matrix
    """
    return np.array(array, dtype=np.float32).reshape((4, 4))


def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat (np.array): a 4x4 homogeneous matrix

    Returns:
        2-tuple:

            - (np.array) (x,y,z) position array in cartesian coordinates
            - (np.array) (x,y,z,w) orientation array in quaternion form
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


@jit_decorator
def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def mat2euler(rmat, axes="sxyz"):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.asarray(rmat, dtype=np.float32)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))


def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


@jit_decorator
def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A
    to a homogenous matrix corresponding to the same point C in frame B.

    Args:
        pose_A (np.array): 4x4 matrix corresponding to the pose of C in frame A
        pose_A_in_B (np.array): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        np.array: 4x4 matrix corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return pose_A_in_B.dot(pose_A)


def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def _skew_symmetric_translation(pos_A_in_B):
    """
    Helper function to get a skew symmetric translation matrix for converting quantities
    between frames.

    Args:
        pos_A_in_B (np.array): (x,y,z) position of A in frame B

    Returns:
        np.array: 3x3 skew symmetric translation matrix
    """
    return np.array(
        [
            0.0,
            -pos_A_in_B[2],
            pos_A_in_B[1],
            pos_A_in_B[2],
            0.0,
            -pos_A_in_B[0],
            -pos_A_in_B[1],
            pos_A_in_B[0],
            0.0,
        ]
    ).reshape((3, 3))


def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Converts linear and angular velocity of a point in frame A to the equivalent in frame B.

    Args:
        vel_A (np.array): (vx,vy,vz) linear velocity in A
        ang_vel_A (np.array): (wx,wy,wz) angular velocity in A
        pose_A_in_B (np.array): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (np.array) (vx,vy,vz) linear velocities in frame B
            - (np.array) (wx,wy,wz) angular velocities in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B.dot(vel_A) + skew_symm.dot(rot_A_in_B.dot(ang_vel_A))
    ang_vel_B = rot_A_in_B.dot(ang_vel_A)
    return vel_B, ang_vel_B


def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Converts linear and rotational force at a point in frame A to the equivalent in frame B.

    Args:
        force_A (np.array): (fx,fy,fz) linear force in A
        torque_A (np.array): (tx,ty,tz) rotational force (moment) in A
        pose_A_in_B (np.array): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (np.array) (fx,fy,fz) linear forces in frame B
            - (np.array) (tx,ty,tz) moments in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T.dot(force_A)
    torque_B = -rot_A_in_B.T.dot(skew_symm.dot(force_A)) + rot_A_in_B.T.dot(torque_A)
    return force_B, torque_B


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.asarray(point[:3], dtype=np.float32)
        M[:3, 3] = point - np.dot(R, point)
    return M


def clip_translation(dpos, limit):
    """
    Limits a translation (delta position) to a specified limit

    Scales down the norm of the dpos to 'limit' if norm(dpos) > limit, else returns immediately

    Args:
        dpos (n-array): n-dim Translation being clipped (e,g.: (x, y, z)) -- numpy array
        limit (float): Value to limit translation by -- magnitude (scalar, in same units as input)

    Returns:
        2-tuple:

            - (np.array) Clipped translation (same dimension as inputs)
            - (bool) whether the value was clipped or not
    """
    input_norm = np.linalg.norm(dpos)
    return (dpos * limit / input_norm, True) if input_norm > limit else (dpos, False)


def clip_rotation(quat, limit):
    """
    Limits a (delta) rotation to a specified limit

    Converts rotation to axis-angle, clips, then re-converts back into quaternion

    Args:
        quat (np.array): (x,y,z,w) rotation being clipped
        limit (float): Value to limit rotation by -- magnitude (scalar, in radians)

    Returns:
        2-tuple:

            - (np.array) Clipped rotation quaternion (x, y, z, w)
            - (bool) whether the value was clipped or not
    """
    clipped = False

    # First, normalize the quaternion
    quat = quat / np.linalg.norm(quat)

    den = np.sqrt(max(1 - quat[3] * quat[3], 0))
    if den == 0:
        # This is a zero degree rotation, immediately return
        return quat, clipped
    else:
        # This is all other cases
        x = quat[0] / den
        y = quat[1] / den
        z = quat[2] / den
        a = 2 * math.acos(quat[3])

    # Clip rotation if necessary and return clipped quat
    if abs(a) > limit:
        a = limit * np.sign(a) / 2
        sa = math.sin(a)
        ca = math.cos(a)
        quat = np.array([x * sa, y * sa, z * sa, ca])
        clipped = True

    return quat, clipped


def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation

    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.asarray(data)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def get_orientation_error(target_orn, current_orn):
    """
    Returns the difference between two quaternion orientations as a 3 DOF numpy array.
    For use in an impedance controller / task-space PD controller.

    Args:
        target_orn (np.array): (x, y, z, w) desired quaternion orientation
        current_orn (np.array): (x, y, z, w) current quaternion orientation

    Returns:
        orn_error (np.array): (ax,ay,az) current orientation error, corresponds to
            (target_orn - current_orn)
    """
    current_orn = np.array([current_orn[3], current_orn[0], current_orn[1], current_orn[2]])
    target_orn = np.array([target_orn[3], target_orn[0], target_orn[1], target_orn[2]])

    pinv = np.zeros((3, 4))
    pinv[0, :] = [-current_orn[1], current_orn[0], -current_orn[3], current_orn[2]]
    pinv[1, :] = [-current_orn[2], current_orn[3], current_orn[0], -current_orn[1]]
    pinv[2, :] = [-current_orn[3], -current_orn[2], current_orn[1], current_orn[0]]
    orn_error = 2.0 * pinv.dot(np.array(target_orn))
    return orn_error


def skew(v):
    """
    Returns the 3x3 skew matrix.
    The skew matrix is a square matrix M{A} whose transpose is also its
    negative; that is, it satisfies the condition M{-A = A^T}.
    @type v: array
    @param v: The input array
    @rtype: array, shape (3,3)
    @return: The resulting skew matrix
    """
    skv = np.roll(np.roll(np.diag(np.asarray(v).flatten()), 1, 1), -1, 0)
    return (skv - skv.T)


def quaternions_orientation_error(target_orn, current_orn):
    """
    Calculates the orientation error between two quaternions
    Qd is the desired orientation
    Qc is the current orientation
    both with respect to the same fixed frame

    return vector part
    """
    ne = current_orn[3]*target_orn[3] + np.dot(current_orn[:3], target_orn[:3])
    ee = current_orn[3]*target_orn[:3] - target_orn[3] * current_orn[:3] + np.dot(skew(current_orn[:3]), target_orn[:3])
    ee *= np.sign(ne)  # disambiguate the sign of the quaternion
    return ee


def limit_quaternion_rotation(q1, q2, max_angle_rad):
    """
    Compute an intermediate quaternion that lies on the shortest path from q1 to q2,
    limited by the maximum allowed rotation angle.

    Parameters:
    q1, q2: numpy arrays of shape (4,) representing quaternions in format [x, y, z, w]
    max_angle_rad: float, maximum allowed rotation angle in radians

    Returns:
    numpy.ndarray: The limited quaternion in [x, y, z, w] format
    """
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate the dot product (note: w component is at index 3)
    dot = np.dot(q1, q2)

    # If the dot product is negative, negate q2
    # This ensures we take the shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Clamp dot product to [-1, 1] for numerical stability
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate the total angle between quaternions
    total_angle = np.arccos(dot)  # Note: This is half the rotation angle

    # If twice the total angle is less than the maximum, return q2
    if 2 * total_angle <= max_angle_rad:
        return q2

    # Calculate the interpolation parameter for the maximum allowed angle
    t = max_angle_rad / (2 * total_angle)

    # Calculate the relative rotation quaternion from q1 to q2
    relative_rot = q2 - q1 * dot
    relative_rot_norm = np.linalg.norm(relative_rot)

    if relative_rot_norm < 1e-7:  # Handle case where quaternions are very close
        return q1

    relative_rot = relative_rot / relative_rot_norm

    # Construct the limited quaternion using the half-angle formula
    half_max_angle = max_angle_rad / 2
    result = q1 * np.cos(half_max_angle) + relative_rot * np.sin(half_max_angle)

    # Ensure the result is normalized
    return result / np.linalg.norm(result)


def get_pose_error(target_pose, current_pose):
    """
    Computes the error corresponding to target pose - current pose as a 6-dim vector.
    The first 3 components correspond to translational error while the last 3 components
    correspond to the rotational error.

    Args:
        target_pose (np.array): a 4x4 homogenous matrix for the target pose
        current_pose (np.array): a 4x4 homogenous matrix for the current pose

    Returns:
        np.array: 6-dim pose error.
    """
    error = np.zeros(6)

    # compute translational error
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos

    # compute rotational error
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (np.cross(r1, r1d) + np.cross(r2, r2d) + np.cross(r3, r3d))

    error[:3] = pos_err
    error[3:] = rot_err
    return error


@jit_decorator
def matrix_inverse(matrix):
    """
    Helper function to have an efficient matrix inversion function.

    Args:
        matrix (np.array): 2d-array representing a matrix

    Returns:
        np.array: 2d-array representing the matrix inverse
    """
    return np.linalg.inv(matrix)


def rotate_2d_point(input, rot):
    """
    rotate a 2d vector counterclockwise

    Args:
        input (np.array): 1d-array representing 2d vector
        rot (float): rotation value

    Returns:
        np.array: rotated 1d-array
    """
    input_x, input_y = input
    x = input_x * np.cos(rot) - input_y * np.sin(rot)
    y = input_x * np.sin(rot) + input_y * np.cos(rot)

    return np.array([x, y])


@jit_decorator
def spd_to_cholesky_vector(spd_matrix):
    """
        Compute Cholesky decomposition for SPD matrix.
        Then, extract and return its lower triangle as a vector.
    """
    cholesky_matrix = np.linalg.cholesky(spd_matrix).ravel()
    tril_mask = np.array([0, 3, 4, 6, 7, 8])
    return cholesky_matrix[tril_mask]


@jit_decorator
def cholesky_vector_to_spd(cholesky_vector):
    """
        Reconstruct Cholesky decomposition matrix from vector.
        Then compute SPD matrix L * L.T
    """
    cholesky_matrix = np.zeros((3, 3), dtype=np.float64)
    mask = np.tril_indices(cholesky_matrix.shape[0], k=0)
    for i in range(6):
        cholesky_matrix[mask[0][i]][mask[1][i]] = cholesky_vector[i]
    return cholesky_matrix @ cholesky_matrix.T


def ortho62axisangle(ortho6):
    return quat2axisangle(ortho62quat(ortho6))


def ortho62quat(ortho6):
    R = ortho62mat(ortho6)
    return mat2quat(R)


def axis_angle2ortho6(axis_angle):
    return quat2ortho6(axisangle2quat(axis_angle))


def quat2ortho6(q):
    return mat2ortho6(quat2mat(q))


def mat2ortho6(R):
    return R[:3, :2].T.flatten()


def ortho62mat(ortho6):
    x_raw, y_raw = ortho6[0:3], ortho6[3:6]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    return np.column_stack((x, y, z))


def force_frame_transform(bTa):
    """
    Calculates the coordinate transformation for force vectors.
    The force vectors obey special transformation rules.
    B{Reference:} Handbook of robotics page 39, equation 2.9
    @type bTa: array, shape (4,4)
    @param bTa: Homogeneous transformation that represents the position
    and orientation of frame M{A} relative to frame M{B}
    @rtype: array, shape (6,6)
    @return: The coordinate transformation from M{A} to M{B} for force
    vectors
    """
    aTb = pose_inv(bTa)
    return motion_frame_transform(aTb).T


def motion_frame_transform(bTa):
    """
    Calculates the coordinate transformation for motion vectors.
    The motion vectors obey special transformation rules.
    B{Reference:} Handbook of robotics page 39, equation 2.9
    @type bTa: array, shape (4,4)
    @param bTa: Homogeneous transformation that represents the position
    and orientation of frame M{A} relative to frame M{B}
    @rtype: array, shape (6,6)
    @return: The coordinate transformation from M{A} to M{B} for motion
    vectors
    """
    bRa = bTa[:3, :3]
    bPa = bTa[:3, 3]
    bXa = np.zeros((6, 6))
    bXa[:3, :3] = bRa
    bXa[3:, :3] = np.dot(_skew_symmetric_translation(bPa), bRa)
    bXa[3:, 3:] = bRa
    return bXa


def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rotate a 3D vector using a quaternion rotation.

    Args:
        vector (np.array): 3D vector [x, y, z]
        quaternion (np.array): Quaternion in format [x, y, z, w]

    Returns:
        np.array: Rotated vector
    """
    # Normalize the quaternion
    q = quaternion / np.linalg.norm(quaternion)
    q_x, q_y, q_z, q_w = q

    # Convert vector to pure quaternion (w = 0)
    v_x, v_y, v_z = vector

    # First multiply: q * v
    temp_w = -q_x*v_x - q_y*v_y - q_z*v_z
    temp_x = q_w*v_x + q_y*v_z - q_z*v_y
    temp_y = q_w*v_y - q_x*v_z + q_z*v_x
    temp_z = q_w*v_z + q_x*v_y - q_y*v_x

    # Second multiply: result * q_conjugate
    q_conjugate = np.array([-q_x, -q_y, -q_z, q_w])

    result_w = -temp_x*(-q_x) - temp_y*(-q_y) - temp_z*(-q_z) + temp_w*q_w
    result_x = temp_w*(-q_x) + temp_y*(-q_z) - temp_z*(-q_y) + temp_x*q_w
    result_y = temp_w*(-q_y) - temp_x*(-q_z) + temp_z*(-q_x) + temp_y*q_w
    result_z = temp_w*(-q_z) + temp_x*(-q_y) - temp_y*(-q_x) + temp_z*q_w

    # Return just the vector part (x, y, z)
    return np.array([result_x, result_y, result_z])


def rotate_by_transformation(error, A_to_B):
    """
    Rotates a 6D spatial vector from one frame to another.

    Args:
        error (np.array): 6D spatial vector [linear (3), angular (3)]
        A_to_B (np.array): 3x3 rotation matrix from frame A to frame B

    Returns:
        np.array: Rotated 6D spatial vector in new frame
    """
    pos_error = A_to_B @ error[:3]
    ori_error = A_to_B @ error[3:]
    return np.concatenate([pos_error, ori_error])


def rotate_quaternion_by_rpy(rpy, q_in, rotated_frame=False):
    """
    if rotated_frame == True, Apply RPY rotation in the reference frame of the quaternion.

    Otherwise, Apply RPY rotation in the rotated frame (the one to which the quaternion has rotated the reference frame).
    """
    q_rot = mat2quat(euler2mat(rpy))

    if rotated_frame:
        q_rotated = quat_multiply(q_in, q_rot)
    else:
        q_rotated = quat_multiply(q_rot, q_in)

    return q_rotated


def compute_pose_error(target_pose, current_pose):
    """
    Computes the pose error between a target pose and a current pose.

    Args:
        target_pose (np.array): Target pose as [position (3), orientation (4)] where orientation is a quaternion (x,y,z,w)
        current_pose (np.array): Current pose as [position (3), orientation (4)] where orientation is a quaternion (x,y,z,w)

    Returns:
        np.array: 6D pose error vector containing [position_error (3), orientation_error (3)]
                  Position error is expressed in the target frame
                  Orientation error is the minimal rotation vector between the two orientations
    """
    relative_distance = np.zeros(6)

    # Compute translational error
    relative_distance[:3] = target_pose[:3] - current_pose[:3]

    # Compute rotational error
    ref_quat = target_pose[3:]
    relative_distance[:3] = rotate_vector_by_quaternion(relative_distance[:3], ref_quat)
    relative_distance[3:] = quaternions_orientation_error(ref_quat, current_pose[3:])

    return relative_distance


def mat2logmap(R):
    """
    Convert a rotation matrix to its logarithm map representation.

    The logarithm map of SO(3) converts a rotation matrix to a 3D vector in the Lie algebra.
    This is useful for representing rotational errors in robotics applications.

    Args:
        R (np.array): 3x3 rotation matrix

    Returns:
        np.array: 3D vector representing the rotation in the Lie algebra (so(3))

    Notes:
        - The output vector represents the rotation as an axis-angle representation
        - The magnitude of the vector is the rotation angle
        - The direction of the vector is the rotation axis
        - For identity rotation, returns zero vector
        - For rotations close to identity, uses small angle approximation
    """
    # Ensure R is a numpy array
    R = np.asarray(R, dtype=np.float64)

    # Check if R is a valid rotation matrix (orthogonal and determinant = 1)
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")

    # Check if R is orthogonal (R * R^T = I)
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError("Input matrix is not orthogonal")

    # Check if determinant is 1 (proper rotation matrix)
    if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Input matrix is not a proper rotation matrix (det != 1)")

    # Extract the skew-symmetric part of the rotation matrix
    # The skew-symmetric part is (R - R^T) / 2
    skew_symmetric = (R - R.T) / 2.0

    # Extract the rotation vector from the skew-symmetric matrix
    # [0, -z, y; z, 0, -x; -y, x, 0] -> [x, y, z]
    rotation_vector = np.array([skew_symmetric[2, 1],
                               skew_symmetric[0, 2],
                               skew_symmetric[1, 0]])

    # Calculate the rotation angle
    # For small rotations, use the magnitude of the rotation vector
    # For larger rotations, use the arcsin of the skew-symmetric part
    angle = np.linalg.norm(rotation_vector)

    # Handle the case where angle is very small (near identity)
    if angle < 1e-6:
        # For very small rotations, the rotation vector is already the log map
        return rotation_vector

    # For larger rotations, we need to scale by the angle
    # The log map is: (angle / sin(angle)) * rotation_vector
    # But we need to handle the case where angle approaches pi

    # Calculate trace of R
    trace_R = np.trace(R)

    # Handle different cases based on the trace
    if trace_R > 2.999:  # Very close to identity
        # Small angle approximation
        return rotation_vector
    elif trace_R < -0.999:  # Rotation close to pi
        # For rotations close to pi, we need special handling
        # Find the eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(R)
        # Find the index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        axis = np.real(eigenvectors[:, idx])

        # Determine the sign of the rotation
        # Check if the rotation is positive or negative around the axis
        test_vector = np.array([1.0, 0.0, 0.0])
        if np.allclose(axis, test_vector):
            test_vector = np.array([0.0, 1.0, 0.0])

        rotated_vector = R @ test_vector
        cross_product = np.cross(axis, test_vector)
        dot_product = np.dot(cross_product, rotated_vector)

        if dot_product > 0:
            angle = np.pi
        else:
            angle = -np.pi

        return angle * axis
    else:
        # Normal case: use the standard formula
        # angle = arccos((trace(R) - 1) / 2)
        cos_angle = (trace_R - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure it's in valid range
        angle = np.arccos(cos_angle)

        # Normalize the rotation vector and scale by angle
        if angle > 1e-6:
            normalized_axis = rotation_vector / np.linalg.norm(rotation_vector)
            return angle * normalized_axis
        else:
            return rotation_vector


def logmap2mat(rotation_vector):
    """
    Convert a rotation vector (logarithm map) back to a rotation matrix.

    This is the inverse of mat2logmap. It converts a 3D vector in the Lie algebra
    back to a rotation matrix using the exponential map.

    Args:
        rotation_vector (np.array): 3D vector representing rotation in Lie algebra

    Returns:
        np.array: 3x3 rotation matrix

    Notes:
        - Uses Rodrigues' rotation formula for efficiency
        - Handles zero rotation vector (returns identity matrix)
    """
    # Ensure input is a numpy array
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)

    if rotation_vector.shape != (3,):
        raise ValueError("Input must be a 3D vector")

    # Calculate the rotation angle
    angle = np.linalg.norm(rotation_vector)

    # Handle zero rotation
    if angle < 1e-6:
        return np.eye(3)

    # Normalize the rotation axis
    axis = rotation_vector / angle

    # Use Rodrigues' rotation formula
    # R = I + sin(angle) * K + (1 - cos(angle)) * K^2
    # where K is the skew-symmetric matrix of the axis

    # Create skew-symmetric matrix K
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # Apply Rodrigues' formula
    R = (np.eye(3) +
         np.sin(angle) * K +
         (1 - np.cos(angle)) * (K @ K))

    return R
