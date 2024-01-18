import copy
import logging
import numpy as np
import threading
import time

from colect.controller.Controller import Controller

_logger = logging.getLogger('colect')


class ParallelPFController(Controller):
    """ParallelPFControllerVelocity class

    A parallel position-force controller that outputs a velocity.

    Args:
        start_position (numpy.ndarray): the initial position
        start_orientation (numpy.ndarray): the initial orientation as an array representing a quaternion
        start_vel (numpy.ndarray): the initial velocity
        start_ft (numpy.ndarray): the initial force-torque vector
    """

    def __init__(self, start_position=np.array([0, 0, 0]), start_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                 start_vel=np.array([0, 0, 0, 0, 0, 0]), start_ft=np.array([0, 0, 0, 0, 0, 0])):
        super().__init__()

        # Declare external inputs
        self.ft_input = start_ft
        self.ft_raw_input = start_ft
        self.pos_input = start_position
        self.rot_input = start_orientation
        self.vel_input = start_vel

        # Output
        self.output = np.array([0, 0, 0, 0, 0, 0])

        # Set controller parameters
        self.Kp = np.zeros((3, 3))
        self.Ki = np.zeros((3, 3))
        self.Kv = np.zeros((3, 3))

        # Targets
        self._p_target = np.array([0, 0, 0])
        self._aa_target = np.array([0, 0, 0])
        self._f_target = np.array([0, 0, 0])
        self._mu_target = np.array([0, 0, 0])

        # Controller States
        self._f_error = np.array([0.0, 0.0, 0.0])
        self._mu_error = np.array([0.0, 0.0, 0.0])
        self._i_f_error = np.array([0.0, 0.0, 0.0])
        self._i_mu_error = np.array([0.0, 0.0, 0.0])

        self._p_error = np.array([0.0, 0.0, 0.0])
        self._i_p_error = np.array([0.0, 0.0, 0.0])
        self._d_p_error = np.array([0.0, 0.0, 0.0])
        self._omega_error = np.array([0.0, 0.0, 0.0])
        self._i_omega_error = np.array([0.0, 0.0, 0.0])
        self._d_omega_error = np.array([0.0, 0.0, 0.0])

        # Initialize controller thread
        self._thread = None
        self._thread_started = False

        # Runtime variables
        self._running = False
        self._pause = False
        
        self._dt = 1 / 500.0 

    def get_output(self):
        return self.output

    def get_p_target(self):
        return self._p_target

    def run_controller(self):
        """ This is the execution thread of parallel position-force controller. Started by start() and
            stopped using stop().
        """

        while self._running:
            if not self._pause:
                v = self.vel_input[:3]
                he = self.ft_raw_input

                f_ext = np.array([he[:3]])
                f_error_unfiltered = self._f_target - f_ext

                force_filtered = self.ft_input
                self._f_error = self._f_target - force_filtered[:3]
                self._i_f_error = (self._i_f_error + f_error_unfiltered * self._dt).flatten()
                
                p_out = self._p_target + self.Kp @ self._f_error + self.Ki @ self._i_f_error - self.Kv @ v
                aa_out = self._aa_target

                self.output = np.concatenate((p_out, aa_out))

        self._thread_started = False

    def start(self):
        """ Start the parallel position-force controller.

        This will spawn a thread for the parallel position-force control.
        """

        self._running = True
        if not self._thread_started:
            self._thread = threading.Thread(target=self.run_controller, daemon=True)
            self._thread.start()
            self._thread_started = True

    def stop(self):
        """Stop the parallel position-force controller

        This function terminates the controller thread.
        """
        self._running = False
        self._thread.join()
        self.reset()
        _logger.info('Parallel Force-Position control stopped!')

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    def reset(self):
        # Reset controller parameters
        self._f_error = np.array([0, 0, 0])
        self._mu_error = np.array([0, 0, 0])
        self._i_f_error = np.array([0, 0, 0])
        self._i_mu_error = np.array([0, 0, 0])

        self.output = np.array([0, 0, 0, 0, 0, 0])

        self._p_target = np.array([0, 0, 0])
        self._aa_target = np.array([0, 0, 0])
        self._f_target = np.array([0, 0, 0])
        self._mu_target = np.array([0, 0, 0])

    def target_error(self):
        return self._p_error

    def set_default_controller_parameters(self):
        self.Kp_f = np.zeros((3, 3))
        self.Ki_f = np.zeros((3, 3))
        self.Kv_f = np.zeros((3, 3))

        self.Kp_p = np.zeros((3, 3))
        self.Ki_p = np.zeros((3, 3))
        self.Kd_p = np.zeros((3, 3))
        self._Kp_aa = np.zeros((3, 3))
        self._Ki_aa = np.zeros((3, 3))
        self._Kd_aa = np.zeros((3, 3))

    def set_target(self, pose, force, torque):
        self.set_target_pose(pose)
        self.set_target_force_torque(force, torque)

    def set_target_force_torque(self, force, torque):
        self._f_target = np.array(force)
        self._mu_target = np.array(torque)

    def set_target_pose(self, pose):
        self._p_target = np.array(pose[0:3])
        self._aa_target = np.array(pose[3:6])

    def ramp_target_force(self, target_force, T):
        _logger.debug("new target: " + str(target_force))
        _logger.debug("old target: " + str(self._f_target))
        start_force = copy.deepcopy(self._f_target)
        ramp = (target_force - start_force) / T
        _logger.debug("ramp: "+str(ramp))
        times = np.arange(start=0, stop=T, step=self._dt)

        def ramp_function():
            for t in times:
                self._f_target = start_force + ramp * t
                time.sleep(self._dt)

        ramp_force_thread = threading.Thread(target=ramp_function, daemon=True)
        ramp_force_thread.start()

        return ramp_force_thread

    def ramp_gains(self, T, Kp=None, Ki=None, Kv=None):
        start_Kp = copy.deepcopy(self.Kp_f)
        ramp_Kp = (Kp - start_Kp) / T if Kp is not None else np.zeros((3, 3))
        start_Ki = copy.deepcopy(self.Ki_f)
        ramp_Ki = (Ki - start_Ki) / T if Ki is not None else np.zeros((3, 3))
        start_Kv = copy.deepcopy(self.Kv_f) if Kv is not None else np.zeros((3, 3))
        ramp_Kv = (Kv - start_Kv) / T
        times = np.arange(start=0, stop=T, step=self._dt)

        def ramp_function():
            for t in times:
                if Kp is not None:
                    self.Kp_f = start_Kp + ramp_Kp * t
                if Ki is not None:
                    self.Ki_f = start_Ki + ramp_Ki * t
                if Kv is not None:
                    self.Kv_f = start_Kv + ramp_Kv * t
                time.sleep(self._dt)

        ramp_gains_thread = threading.Thread(target=ramp_function, daemon=True)
        ramp_gains_thread.start()

        return ramp_gains_thread

