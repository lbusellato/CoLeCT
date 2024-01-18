# -*- coding: utf-8 -*-
import argparse
import copy
import sys
import logging
import time
import numpy as np
import quaternion

from os.path import join, dirname, abspath
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d.transform_manager import TransformManager

from colect.datatypes import Quaternion
from colect.robot.ur_robot import URRobot
from colect.utils.math import *
from colect.controller.AdmittanceController import AdmittanceController

_logger = logging.getLogger('colect')


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

ROOT = dirname(dirname(abspath(__file__)))

approach_position = np.array([-0.365, 0.290, 0.212])
tcp_orientation = np.array([0.0,  0.978, -0.208,  0.0])
start_position = np.array([-0.365, 0.290, 0.04])

APPROACH_POSE = np.concatenate((approach_position, Quaternion.from_array(tcp_orientation).as_rotation_vector()))
START_POSE = np.concatenate((start_position, Quaternion.from_array(tcp_orientation).as_rotation_vector()))

def main():
    #robot_ip = "172.17.0.2"
    robot_ip = "192.168.1.11"
    setup_logging(logging.DEBUG)
    frequency = 500.0  # Hz
    dt = 1 / frequency

    # Initialize transform manager
    tm = TransformManager()

    _logger.info("Initializing robot...")
    ur_robot = URRobot(robot_ip)

    ur_robot.moveJ(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))
    time.sleep(0.5)
    ur_robot.moveJ_IK(APPROACH_POSE)
    time.sleep(0.5)
    ur_robot.receive_data()
    time.sleep(0.5)
    ur_robot.zero_ft_sensor()
    time.sleep(0.5)
    ur_robot.moveJ_IK(START_POSE)
    time.sleep(0.5)
    input("Press any key to start")

    print("1")
    # receive initial robot data
    ur_robot.receive_data()

    # get current robot pose
    T_base_tcp = get_robot_pose_as_transform(ur_robot)

    # define a tip wrt. the TCP frame
    quat_identity = quaternion.as_float_array(quaternion.quaternion(1.0, 0.0, 0.0, 0.0))
    T_tcp_tip = pt.transform_from_pq(np.hstack((np.array([0, 0, 0]), quat_identity)))
    T_tip_tcp = np.linalg.inv(T_tcp_tip)
    T_base_tip = T_base_tcp @ T_tcp_tip

    # get tip in base as position and quaternion
    T_base_tip_pq = pt.pq_from_transform(T_base_tip)
    T_base_tip_pos_init = T_base_tip_pq[0:3]
    T_base_tip_quat_init = T_base_tip_pq[3:7]

    x_desired = start_position
    T_base_tip_circle = pt.transform_from_pq(np.hstack((x_desired, T_base_tip_quat_init)))
    T_base_tcp_circle = T_base_tip_circle @ T_tip_tcp

    # Use moveL to move to the initial point on the circle.
    ur_robot.moveL(get_transform_as_ur_pose(T_base_tcp_circle))

    _logger.info("Initializing AdmittanceController...")
    adm_controller = AdmittanceController(start_position=x_desired, start_orientation=T_base_tip_quat_init,
                                                  start_ft=ur_robot.ft)
    _logger.info("Starting AdmittanceController!")

    # The controller parameters can be changed, the following parameters corresponds to the default,
    # and we set them simply to demonstrate that they can be changed.
    adm_controller.M = np.diag([2.5, 2.5, 2.5])
    adm_controller.D = np.diag([500, 500, 100]) #500, 500, 700
    adm_controller.K = np.diag([5*54, 5*54, 0.0])

    adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
    adm_controller.Do = np.diag([150.0, 150.0, 100.0])
    adm_controller.Ko = np.diag([50, 50, 50])

    f_target = np.array([0, 0, -5])


    try:
        while True:
            #start_time = ur_robot.initPeriod()
            start_time = time.time()

            ur_robot.receive_data()

            # get current robot pose
            T_base_tcp = get_robot_pose_as_transform(ur_robot)
            T_base_tip = T_base_tcp @ T_tcp_tip

            # get tip in base as position and quaternion
            T_base_tip_pq = pt.pq_from_transform(T_base_tip)
            T_base_tip_pos = T_base_tip_pq[0:3]
            T_base_tip_quat = T_base_tip_pq[3:7]

            # get current robot force-torque in base (measured at tcp)
            f_base = ur_robot.ft[0:3]
            mu_base = ur_robot.ft[3:6]

            # rotate forces from base frame to TCP frame (necessary on a UR robot)
            R_tcp_base = np.linalg.inv(T_base_tcp[:3, :3])
            f_tcp = R_tcp_base @ f_base
            mu_tcp = R_tcp_base @ mu_base

            # use wrench transform to place the force torque in the tip.
            mu_tip, f_tip = wrench_trans(mu_tcp, f_tcp, T_tcp_tip)

            # rotate forces back to base frame
            R_base_tip = T_base_tip[:3, :3]
            f_base_tip = R_base_tip @ f_tip

            # the input position and orientation is given as tip in base
            adm_controller.pos_input = T_base_tip_pos
            adm_controller.rot_input = quaternion.from_float_array(T_base_tip_quat)
            adm_controller.q_input = ur_robot.getActualQ()
            adm_controller.ft_input = np.hstack((f_base_tip + f_target, mu_tip))

            adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(T_base_tip_quat_init))

            # step the execution of the admittance controller
            adm_controller.step()
            output = adm_controller.get_output()
            output_position = output[0:3]
            output_quat = output[3:7]

            # rotate output from tip to TCP before sending it to the robot
            T_base_tip_out = pt.transform_from_pq(np.hstack((output_position, output_quat)))
            T_base_tcp_out = T_base_tip_out @ T_tip_tcp
            base_tcp_out_ur_pose = get_transform_as_ur_pose(T_base_tcp_out)

            # set position target of the robot
            ur_robot.pos_target = base_tcp_out_ur_pose
            ur_robot.control_step()
            end_time = time.time()
            cycle_time = start_time - end_time
            if cycle_time < 0.002:
                time.sleep(0.002-cycle_time)
            #ur_robot.waitPeriod(start_time)
    except KeyboardInterrupt:
        pass
    #adm_controller.stop()
    ur_robot.stop_control()

if __name__ == "__main__":
    main()
