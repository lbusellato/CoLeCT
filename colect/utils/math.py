#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d.transform_manager import TransformManager
import quaternion
import math
import sys
import random


def get_transform_as_ur_pose(transform):
    T_base_tcp_out_pq = pt.pq_from_transform(transform)
    T_base_tcp_out_pos = T_base_tcp_out_pq[0:3]
    T_base_tcp_out_rotvec = pr.compact_axis_angle_from_quaternion(T_base_tcp_out_pq[3:7])
    return np.concatenate((T_base_tcp_out_pos, T_base_tcp_out_rotvec))


def get_transform_as_pq(tm, from_frame, to_frame):
    T_base_tip = tm.get_transform(from_frame, to_frame)
    T_base_tip_pq = pt.pq_from_transform(T_base_tip)
    T_base_tip_pos = T_base_tip_pq[0:3]
    T_base_tip_quat = T_base_tip_pq[3:7]
    return T_base_tip_pos, T_base_tip_quat


def get_robot_pose_as_transform(robot):
    pq = np.hstack((robot.position, quaternion.as_float_array(robot.rotation)))
    T_base_tcp = pt.transform_from_pq(pq)
    return T_base_tcp


def rotvec_to_R(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rotvec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0] * r[0], r[0] * r[1], r[0] * r[2]],
            [r[1] * r[0], r[1] * r[1], r[1] * r[2]],
            [r[2] * r[0], r[2] * r[1], r[2] * r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat


def R_to_rotvec(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation vector with angle multiplied onto axis.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = math.atan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return np.array([axis[0] * theta, axis[1] * theta, axis[2] * theta])

def skew_symmetric(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    Sv = np.zeros((3, 3))
    Sv[1, 0] = z
    Sv[2, 0] = -y
    Sv[0, 1] = -z
    Sv[2, 1] = x
    Sv[0, 2] = y
    Sv[1, 2] = -x
    return Sv


def adjoint(matrix):
    # Assumes input is 4x4 transformation matrix
    R_mat = matrix[0:3, 0:3]
    p = matrix[0:3, 3]

    adj_T = np.zeros((6, 6))
    adj_T[0:3, 0:3] = R_mat
    adj_T[3:6, 0:3] = skew_symmetric(p) @ R_mat
    adj_T[3:6, 3:6] = R_mat

    return adj_T


def wrench_trans(torques, forces, T):
    """
    Transforms the input wrench (torques, forces) with T.
    Outputs transformed wrench (torques, forces)
    """
    wrench_in_A = np.hstack((torques, forces))
    wrench_in_B = adjoint(T).T @ wrench_in_A
    return wrench_in_B[:3], wrench_in_B[3:]


def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()

def gcd(a, b):
    if (a < b):
        return gcd(b, a)

        # base case
    if abs(b) < 1e-5:
        return a
    else:
        return gcd(b, a - math.floor(a / b) * b)


def lcm(a, b=None):
    if b is not None:
        return (a * b) / gcd(a, b)
    else:
        out = lcm(a[0], a[1])
        for i in range(2, len(a)):
            out = lcm(out, a[i])

        return out


def random_quaternion():
    z = 100000
    while z > 1.0:
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        z = x * x + y * y
    w = 100000
    while w > 1.0:
        u = random.uniform(-1.0, 1.0)
        v = random.uniform(-1.0, 1.0)
        w = u * u + v * v

    s = np.sqrt((1 - z) / w)
    return quaternion.quaternion(x, y, s * u, s * v)


def normalize_vector(v):
    return v/np.linalg.norm(v)


def calculate_rotation_between_vectors(v_from, v_to):
    v_from = normalize_vector(v_from)
    v_to = normalize_vector(v_to)
    dotp = max(min(np.dot(v_from, v_to), 1.0), -1.0)
    angle = np.arccos(dotp)
    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)

    #print("axis" + str(axis))
    #print("angle" + str(angle))

    if axis_norm < 1e-5:
        if angle < 1e-5:
            #print("0 deg")
            return np.identity(3)
        elif np.pi - angle < 1e-5:
            # Choose any vector orthogonal to x_world
            #("180 deg")
            # axis = np.array([0, 1, 0])
            axis = arbitrary_orthogonal_vector(v_from)
            #print("new axis: " + str(axis))
            axis_norm = np.linalg.norm(axis)

    axis /= axis_norm

    r = R.from_rotvec(axis * angle)
    Rv = r.as_matrix()

    #print(Rv)
    return Rv


def arbitrary_orthogonal_vector(vec):
    Ax = np.abs(vec[0])
    Ay = np.abs(vec[1])
    Az = np.abs(vec[2])
    if Ax < Ay:
        P = np.array([0, -vec[2], vec[1]]) if Ax < Az else np.array([-vec[1], vec[0], 0])
    else:
        P = np.array([vec[2], 0, -vec[0]]) if Ay < Az else np.array([-vec[1], vec[0], 0])

    return P


def quat_to_axang(q):
    for i in range(1, len(q)):
        if np.dot(q[i].vec, q[i-1].vec) < 0:
            q[i] *= -1
    return quaternion.as_float_array(2*np.log(np.normalized(q)))[..., 1:]


def rotate_ur_wrench_to_tcp(rots_aa, forces_in_base, torques_in_base):
    if rots_aa.ndim == 1 and forces_in_base.ndim == 1 and torques_in_base.ndim == 1:
        rot = R.from_rotvec(rots_aa)

        # Assuming UR only rotates to the base frame
        R_tcp_base = np.linalg.inv(rot.as_matrix())
        torque_in_tcp = R_tcp_base @ torques_in_base
        force_in_tcp = R_tcp_base @ forces_in_base
        return force_in_tcp, torque_in_tcp
    else:
        forces_in_tcp = []
        torques_in_tcp = []

        for rot_aa, force_in_base, torque_in_base in zip(rots_aa, forces_in_base, torques_in_base):
            rot = R.from_rotvec(rot_aa)

            # Assuming UR only rotates to the base frame
            R_tcp_base = np.linalg.inv(rot.as_matrix())
            torque_in_tcp = R_tcp_base @ torque_in_base
            force_in_tcp = R_tcp_base @ force_in_base
            forces_in_tcp.append(force_in_tcp)
            torques_in_tcp.append(torque_in_tcp)

        return np.array(forces_in_tcp), np.array(torques_in_tcp)


def axis_angles_to_quaternions(axis_angles):
    # Ensure that the orientations are formatted properly
    #for i in range(1, len(axis_angles)):
    #    if np.dot(axis_angles[i], axis_angles[i - 1]) < 0:
    #        axis_angles[i] *= -1

    q = quaternion.from_rotation_vector(axis_angles)

    # Ensure that the quaternions do not flip sign
    for i in range(1, len(q)):
        q0 = q[i - 1]
        q1 = q[i]

        if np.dot(q0.vec, q1.vec) < 0:
            q[i] *= -1

    return q

def current2torque(current, torque_constants):
    # placeholder conversion function
    # K = np.diag(torque_constants)
    # out = current / K
    # out = np.divide(current, K)

    # out = np.array(current).flatten() / np.array(torque_constants).flatten()
    out = np.array(current).flatten() * np.array(torque_constants).flatten()

    return out

def get_param_as_matrix(val, dim=3):
    if isinstance(val, list):
        if len(val) == dim:
            return np.diag(val)
        else:
            raise TypeError("Wrong list size specified expected length to be "+str(dim)+" got: "+str(len(val)))
    elif isinstance(val, np.ndarray):
        if val.shape == (dim,):
            return np.diag(val)
        elif val.shape == (dim, dim):
            return val
        else:
            raise TypeError("Wrong input shape specified, expected ("+str(dim)+","+str(dim)+") or ("+str(dim)+",) got: "+str(val.shape))
    else:
        raise TypeError("Wrong input type specified, expected list, numpy array or numpy matrix")


if __name__ == "__main__":
    R_mat = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    p = np.array([[0, 0, 0.1]]).T
    T = np.hstack((R_mat, p))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    #print("Trans\n", T)

    torques = np.asarray([0, 0, 0])
    forces = np.asarray([1, 0, 0])

    torques_out, forces_out = wrench_trans(torques, forces, T)

    #print(f"Torques before \t {torques}")
    print(f"Torques after \t {torques_out}")

    print(f"Forces before \t {forces}")
    print(f"Forces after \t {forces_out}")
