from colect.controller.Controller import Controller
from colect.utils.math import skew_symmetric
from colect.utils.math import get_param_as_matrix
import roboticstoolbox as rtb
import time
import math
import logging
import threading
import quaternion
import numpy as np
from scipy.linalg import block_diag
import quaternion

_logger = logging.getLogger('colect')


# define potential field logistic function
def l_func(d, d_minus=0.1, d_plus=0.7):
    # define smoothness of transition with sharpness parameter k, defined as:
    k = 12 / (d_plus - d_minus)
    return 1 / (1 + np.exp(k * (d - (d_minus + d_plus / 2))))


class AdmittanceController(Controller):
    """ AdmittanceControllerPosition class

    An admittance controller that outputs a position.

    Args:
        start_position (numpy.ndarray): the initial position
        start_orientation (numpy.ndarray): the initial orientation as an array representing a quaternion
        start_ft (numpy.ndarray): the initial force-torque vector
        singularity_avoidance (Bool): whether to perform singularity avoidance or not
    """

    def __init__(self, start_position=np.array([0, 0, 0]), start_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                 start_ft=np.array([0, 0, 0, 0, 0, 0]), start_q=np.array([0, 0, 0, 0, 0, 0]),
                 singularity_avoidance=False):
        super().__init__()

        # Declare external inputs
        self.ft_input = start_ft
        self.pos_input = start_position
        self.q_input = start_q
        self.rot_input = start_orientation

        # controller output target, initialized to start pose
        self.output = np.concatenate((start_position, start_orientation))

        # Specification of controller parameters
        self.M = np.diag([22.5, 22.5, 22.5])  # Positional Mass
        self.K = np.diag([0, 0, 0])  # Positional Stiffness
        self.D = np.diag([85, 85, 85])  # Positional Damping

        self.Mo = np.diag([0.25, 0.25, 0.25])  # Orientation Mass
        self.Ko = np.diag([0, 0, 0])  # Orientation Stiffness
        self.Do = np.diag([5, 5, 5])  # Orientation Damping

        # Initialize desired frame
        self._x_desired = start_position
        self._quat_desired = quaternion.from_float_array(start_orientation)

        # Compliance frame initialization
        self._dx_c = np.array([0.0, 0.0, 0.0])
        self._omega_c = np.array([0.0, 0.0, 0.0])
        self._quat_c = quaternion.from_float_array(start_orientation)

        # Error terms
        self._x_e = np.array([0.0, 0.0, 0.0])
        self._dx_e = np.array([0.0, 0.0, 0.0])
        self._omega_e = np.array([0.0, 0.0, 0.0])
        self._quat_e = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

        # Robot model (needed for singularity avoidance)
        self._robot_model = rtb.models.DH.UR10()

        # singularity avoidance force-torque max limits (wrist-lock)
        self._K_w = np.diag([0, 0, 0, 4, 4, 4])
        # singularity avoidance force-torque max limits (head-lock)
        self._K_h = np.diag([45, 45, 0, 0, 0, 0])
        # define constant for pushback-force (elbow-lock)
        self._k_3 = 85
        # define position threshold for joint 3 (elbow-lock)
        self._t_3 = 0.8  # 1.2 # rad

        self.perform_singularity_avoidance = singularity_avoidance

    def step(self):
        """
        Step the execution of the admittance controller (must be called in a loop externally)
        """
        f_base = self.ft_input[0:3]
        mu_desired = self.ft_input[3:6]

        # --- Perform singularity avoidance calculations ---
        if self.perform_singularity_avoidance:
            actual_q = self.q_input

            # WRIST
            q_5 = actual_q[4]
            # calculate distance to wrist singularity.
            d_w = math.sin(q_5) ** 2
            # calculate gradient of the distance function
            nabla_d_w = np.array([0, 0, 0, 0, math.sin(2 * q_5), 0]).T
            # calculate manipulator jacobian in the world frame
            J = self._robot_model.jacob0(np.array(actual_q))
            # calculate the repulsive force-torque away from wrist-lock
            ft_w = self._K_w * J @ (nabla_d_w / np.linalg.norm(nabla_d_w)) * l_func(d_w)
            f_wrist = ft_w[0:3]
            tau_wrist = ft_w[3:6]

            # ELBOW
            q_3 = actual_q[2]
            ee_trans = self.pos_input[0:3]
            norm_ee_trans = ee_trans / np.linalg.norm(ee_trans)

            # calculate pushback force for elbow-lock
            if q_3 > 0:
                if q_3 < self._t_3:
                    f_elbow = norm_ee_trans * self._k_3 * (q_3 - self._t_3)
                else:
                    f_elbow = np.zeros(3)
            else:
                if q_3 > -self._t_3:
                    f_elbow = -norm_ee_trans * self._k_3 * (q_3 + self._t_3)
                else:
                    f_elbow = np.zeros(3)

            # HEAD
            x = self.pos_input[0]
            y = self.pos_input[1]
            # calculate distance to head singularity.
            d_h = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
            # calculate gradient of the distance function
            nabla_d_h = np.array([2 * x, 2 * y, 0, 0, 0, 0]).T
            # calculate the repulsive force away from head-lock
            f_h = self._K_h @ (nabla_d_h / (2 * math.sqrt(math.pow(x, 2) + math.pow(y, 2)))) * l_func(d_h, 0.15, 0.3)
            f_head = f_h[0:3]

            # Combined singularity avoidance forces and torques
            f_singu_comb = f_wrist + f_elbow + f_head
            tau_singu_comb = tau_wrist
        else:
            # If singularity avoidance is disabled set avoidance forces to zero.
            f_singu_comb = np.array([0.0, 0.0, 0.0])
            tau_singu_comb = np.array([0.0, 0.0, 0.0])

        # --- Compute positional part of "compliance frame" x_c ---
        # compute acceleration error of compliance frame
        ddx_e = np.linalg.inv(self.M) @ ((f_base + f_singu_comb) - self.K @ self._x_e - self.D @ self._dx_e)
        # integrate to get velocity error
        self._dx_e += ddx_e * self._dt
        # integrate to get position error
        self._x_e += self._dx_e * self._dt
        # add desired position
        x_c = self._x_desired + self._x_e

        # --- Compute rotational part of "compliance frame" quat_c ---
        # compute angular acceleration error of compliance frame
        E = self._quat_e.w * np.eye(3) - skew_symmetric(self._quat_e.imag)
        Ko_mark = 2 * E.T @ self.Ko
        domega_e = np.linalg.inv(self.Mo) @ ((mu_desired - tau_singu_comb) - Ko_mark @ self._quat_e.imag - self.Do @ self._omega_e)

        # integrate to get angular velocity error
        self._omega_e += domega_e * self._dt

        # integrate to get quaternion for orientation
        half_omega_e_dt = 0.5 * self._omega_e * self._dt
        omega_quat = np.exp(quaternion.quaternion(0, half_omega_e_dt[0], half_omega_e_dt[1], half_omega_e_dt[2]))
        self._quat_e = omega_quat * self._quat_e
        # multiply with the desired quaternion
        self._quat_c = self._quat_desired * self._quat_e
        quat_c_arr = quaternion.as_float_array(self._quat_c)
        # combine position and rotational part of compliance frame for the output
        self.output = [x_c[0], x_c[1], x_c[2], quat_c_arr[0], quat_c_arr[1], quat_c_arr[2], quat_c_arr[3]]

    def get_output(self):
        return self.output

    def reset(self):
        self._dx_c = np.array([0.0, 0.0, 0.0])
        self._omega_c = np.array([0.0, 0.0, 0.0])
        self._quat_c = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

        # Error terms
        self._x_e = np.array([0.0, 0.0, 0.0])
        self._dx_e = np.array([0.0, 0.0, 0.0])
        self._omega_e = np.array([0.0, 0.0, 0.0])
        self._quat_e = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    def set_default_controller_parameters(self):
        """ A convenience function for setting the control parameters to their default values.

        The function uses the default values for setting mass, stiffness and damping.
        """
        self.M = np.diag([22.5, 22.5, 22.5])
        self.K = np.diag([0, 0, 0])
        self.D = np.diag([85, 85, 85])

        self.Mo = np.diag([0.25, 0.25, 0.25])
        self.Ko = np.diag([0, 0, 0])
        self.Do = np.diag([5, 5, 5])

    def set_desired_frame(self, position, quat):
        """ Set the desired frame for the admittance controller.

        Specify a desired position, and orientation (as a quaternion)

        Args:
            position (numpy.ndarray): The desired position [x, y, z]
            quat (quaternion.quaternion): The desired orientation as quaternion eg. quaternion.quaternion(1, 0, 0, 0)

        """
        self._x_desired = position
        self._quat_desired = quat
