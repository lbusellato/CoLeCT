import copy
import joblib
import logging
import numpy as np
import quaternion
import sys
import time

from os.path import join, dirname, abspath
from pytransform3d import transformations as pt

from colect.datatypes import Quaternion
from colect.robot import URRobot, DataRecording
from colect.utils.math import *
from colect.utils.Timing import Timing
from colect.utils import linear_traj, force_step
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

ROOT = dirname(dirname(dirname(abspath(__file__))))

def main():
    
    robot_ip = "192.168.1.11"

    setup_logging(logging.INFO)

    frequency = 500.0  # Hz
    dt = 1 / frequency

    _logger.info("Initializing robot...")
    ur_robot = URRobot(robot_ip, frequency=frequency)

    timing = Timing()

    # Load the trained model and trajectory
    kmp = joblib.load(join(ROOT, "trained_models", "experiment5_kmp.mdl"))
    kmp.verbose = False

    # New trajectory for KMP
    start_pose = np.array([-0.365, 0.29, 0.128])
    end_pose = np.array([-0.395, 0.29, 0.128])

    first_slide = linear_traj(start_pose, end_pose, n_points=3*20).T

    first_push = np.tile(first_slide[:,-1], (3*15, 1)).T

    start_pose = np.array([-0.395, 0.29, 0.128])
    end_pose = np.array([-0.415, 0.29, 0.128])
    second_slide = linear_traj(start_pose, end_pose, n_points=3*30).T
    
    second_push = np.tile(second_slide[:,-1], (3*15, 1)).T
    traj = np.hstack((first_slide, first_push, second_slide, second_push))

    kmp_force_target, _ = kmp.predict(traj)
    #kmp_force_target = force_step(-5, -10, 250, 85)

    rot_vector = np.array([3.14, 0.0, 0.0])
    rotvecs = np.tile(rot_vector, (traj.shape[1], 1)).T
    traj = np.vstack((traj, rotvecs))
    traj_i = 0

    START_POSE = traj[:,0]
    APPROACH_POSE = traj[:,0] + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])

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
    input("Press any key to begin")

    # receive initial robot data
    ur_robot.receive_data()

    # get current robot pose
    T_base_tcp = get_robot_pose_as_transform(ur_robot)

    # get tip in base as position and quaternion
    T_base_tcp_pq = pt.pq_from_transform(T_base_tcp)
    T_base_tcp_pos_init = T_base_tcp_pq[0:3]
    T_base_tcp_quat_init = T_base_tcp_pq[3:7]

    _logger.info("Initializing AdmittanceController...")
    adm_controller = AdmittanceController(start_position=T_base_tcp_pos_init, start_orientation=T_base_tcp_quat_init,
                                                  start_ft=ur_robot.ft)
    _logger.info("Starting AdmittanceController!")

    # Controller gains
    #adm_controller.M = np.diag([2.5, 2.5, 2.5])
    #adm_controller.D = np.diag([500, 500, 100]) 
    #adm_controller.K = np.diag([5*54, 5*54, 0.0])
    adm_controller.M = np.diag([2.5, 2.5, 2.5])
    adm_controller.D = np.diag([500, 500, 200]) 
    adm_controller.K = np.diag([5*54, 5*54, 0])

    adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
    adm_controller.Do = np.diag([150.0, 150.0, 100.0])
    adm_controller.Ko = np.diag([50, 50, 50])

    # Feedforward force terms -> force target
    f_target = np.array([0, 0, 0])

    # Set up recording
    recording_dir = join(ROOT, "reproductions")
    recording_filename = "recording_" + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    recorder = DataRecording(robot_ip=robot_ip, directory=recording_dir, filename=recording_filename, frequency=frequency)
    recorder.add_data_labels(['target_TCP_force_0',
                              'target_TCP_force_1',
                              'target_TCP_force_2',
                              'target_TCP_force_3',
                              'target_TCP_force_4',
                              'target_TCP_force_5',
                              'kmp_target_TCP_pose_0',
                              'kmp_target_TCP_pose_1',
                              'kmp_target_TCP_pose_2',
                              'kmp_target_TCP_pose_3',
                              'kmp_target_TCP_pose_4',
                              'kmp_target_TCP_pose_5',
                              'us_image_timestamp'])
    recorder.start()

    do_homing = True
    done = False
    elapsed = 0
    duration = 20.0
    try:
        while not done:
            start_time = time.perf_counter()

            ur_robot.receive_data()

            if np.any(np.abs(ur_robot.ft) > 25):
                ur_robot.stop_control()
                print("F too high! Stopping...")
                done = True

            # Get current robot pose
            T_base_tcp = get_robot_pose_as_transform(ur_robot)

            # Get tip in base as position and quaternion
            T_base_tcp_pq = pt.pq_from_transform(T_base_tcp)
            T_base_tcp_pos = T_base_tcp_pq[0:3]
            T_base_tcp_quat = T_base_tcp_pq[3:7]

            # Filter force torque in base measured at TCP
            f_base = ur_robot.ft[0:3]
            mu_base = ur_robot.ft[3:6]

            x_desired = traj[:3, traj_i]
            f_target = np.array([0.0, 0.0, kmp_force_target[2, traj_i]])


            # The input position and orientation is given as tip in base
            adm_controller.pos_input = T_base_tcp_pos
            adm_controller.rot_input = quaternion.from_float_array(T_base_tcp_quat)
            adm_controller.q_input = ur_robot.getActualQ()
            adm_controller.ft_input = np.hstack((f_base + f_target, mu_base))

            adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(T_base_tcp_quat_init))

            # Step the execution of the admittance controller
            adm_controller.step()
            output = adm_controller.get_output()
            output_position = output[0:3]
            output_quat = output[3:7]

            T_base_tcp_out = pt.transform_from_pq(np.hstack((output_position, output_quat)))
            base_tcp_out_ur_pose = get_transform_as_ur_pose(T_base_tcp_out)

            # Set position target of the robot
            ur_robot.pos_target = base_tcp_out_ur_pose
            ur_robot.control_step()

            # Record data
            recorder.add_data(np.hstack((f_target, mu_base, x_desired, rotvecs[:,traj_i], timing.get_timestamp())))

            # Next traj point, end if there isn't one
            if elapsed > duration / traj.shape[1]:
                elapsed = 0
                traj_i += 1
                if traj_i >= traj.shape[1]:
                    done = True

            # Ensure constant cycle time
            end_time = time.perf_counter()
            cycle_time = end_time - start_time
            elapsed += cycle_time
            if cycle_time < dt:
                time.sleep(dt - cycle_time)
    except KeyboardInterrupt:
        # CTRL-C caught, don't move the robot further
        do_homing = False

    # Stop the recording
    recorder.stop()

    # Stop the robot
    ur_robot.stop_control()

    if do_homing:
        # Return to the home pose
        time.sleep(0.5)
        ur_robot.moveJ(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))
        time.sleep(0.5)
        
    recorder.verify_data_integrity(max_num_of_missed_samples=100)

if __name__ == "__main__":
    main()