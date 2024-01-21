from colect.robot.robot import Robot
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl

import time
import logging
import threading
import quaternion
from enum import IntEnum
import numpy as np


_logger = logging.getLogger('sdu_controllers')


class URRobot(Robot, RTDEControl, RTDEReceive):
    """ URRobot class

    Args:
        ip (str) : IP-address of robot
    """

    class RuntimeState(IntEnum):
        STOPPING = 0,
        STOPPED = 1,
        PLAYING = 2,
        PAUSING = 3,
        PAUSED = 4,
        RESUMING = 5

    class ControlType(IntEnum):
        POSITION = 0,
        VELOCITY = 1,
        TORQUE = 2

    def __init__(self, ip, frequency=500.0):
        Robot.__init__(self, ip)
        self._frequency = frequency
        self._dt = 1.0 / self._frequency
        RTDEControl.__init__(self, self._ip)
        RTDEReceive.__init__(self, self._ip, self._frequency, [])

        # self._rtde_c = RTDEControl(self._ip)
        # self._rtde_r = RTDEReceive(self._ip, frequency, [])

        # Servo options
        self._servo_vel = 0.0  # Not used currently
        self._servo_acc = 0.0  # Not used currently
        self._servo_p_gain = 0.03  # proportional gain
        self._servo_lookahead_t = 1000  # lookahead time

        self._deceleration_rate = 20.0  # m/s^2
        self._vel_tool_acceleration = 1.0  # 1.4  # m/s^2

        # Init robot targets
        self.pos_target = self.getActualTCPPose()
        self.vel_target = [0, 0, 0, 0, 0, 0]
        self.torque_target = [0, 0, 0, 0, 0, 0]

        # Receive data thread is started automatically
        self._ctrl_type = self.ControlType.POSITION

    def receive_data(self):
        """
        receive_data function (must be called in a loop externally)
        """
        self.velocity_ = np.array(self.getActualTCPSpeed())
        actual_tcp_pose = self.getActualTCPPose()
        self.position_ = np.array(actual_tcp_pose[0:3])
        self.rotation_ = quaternion.from_rotation_vector(np.array(actual_tcp_pose[3:6]))
        self.rotation_vector_ = np.array(actual_tcp_pose[3:6])
        self.ft_ = self.getActualTCPForce()

    def get_state(self, time = 0):
       return np.concatenate(([time], 
                              self.position,
                              self.velocity,
                              self.rotation_vector,
                              self.ft))

    def control_step(self):
        """
        Perform robot control step (must be called in a loop externally)
        """
        success = False
        if self._ctrl_type == self.ControlType.POSITION:
            success = self.servoL(self.pos_target, self._servo_vel, self._servo_acc, self._dt, self._servo_p_gain,
                                  self._servo_lookahead_t)
        elif self._ctrl_type == self.ControlType.VELOCITY:
            success = self.speedL(self.vel_target, self._vel_tool_acceleration)
        elif self._ctrl_type == self.ControlType.TORQUE:
            success = self.jointTorque(self.torque_target)

        if not success:
            runtime_state = self.getRuntimeState()
            if runtime_state == self.RuntimeState.STOPPED:
                _logger.info("Command did not succeed, protective, emergency or program stopped, restart control!")

    def set_control_type(self, control_type=ControlType.POSITION):
        self._ctrl_type = control_type

    def stop_control(self):
        self.reset_target()

        if self._ctrl_type == self.ControlType.POSITION:
            self.servoStop(self._deceleration_rate)
        elif self._ctrl_type == self.ControlType.VELOCITY:
            self.speedStop(self._deceleration_rate)
        elif self._ctrl_type == self.ControlType.TORQUE:
            self.torqueStop()

    def reset_target(self):
        self.pos_target = self.getActualTCPPose()
        self.vel_target = [0, 0, 0, 0, 0, 0]
        self.torque_target = [0, 0, 0, 0, 0, 0]

    def zero_ft_sensor(self):
        self.zeroFtSensor()
        time.sleep(0.2)
