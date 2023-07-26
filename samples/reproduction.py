import rtde_control
import rtde_receive
import numpy as np
import time

from control import lqr
from os.path import join, dirname, abspath, exists
from src.hex12 import HEX12

ROOT = dirname(dirname(abspath(__file__)))

def symmetrize(matrices):
    for i in range(matrices.shape[-1]):
        matrix = matrices[:,:,i]
        matrices[:,:,i] = (matrix + matrix.T) / 2
    return matrices

def to_axis_angle(rotations):
    axis_angle = np.zeros((3, rotations.shape[-1]))
    for i in range(rotations.shape[-1]):    
        quaternion = np.array(rotations[:,i])
        # Extract the angle of rotation
        angle = 2*np.arccos(quaternion[0])  # Angle in radians
        # Avoid division by zero for small angles
        if np.abs(angle) > 1e-8:
            # Extract the axis of rotation
            axis = quaternion[1:] / np.sin(angle/2)
            axis_angle[:,i] = (angle)*axis
    return axis_angle

current_force = np.zeros(3)
def hex12_callback(wrench) -> None:
    current_force = wrench[:3]

def main():    
    rtde_c = rtde_control.RTDEControlInterface("192.168.1.11")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")

    # Pull the KMP trajectories
    pos = np.load(join(ROOT, 'trained_models/mu_pos_kmp.npy'))
    if not exists(join(ROOT, 'controller/mu_rot_kmp_axis_angle.npy')):
        rot = np.load(join(ROOT, 'trained_models/mu_rot_kmp_quats.npy'))
        rot = to_axis_angle(rot)
        np.save(join(ROOT, 'controller/mu_rot_kmp_axis_angle.npy'), rot)
    else:
        rot = np.load(join(ROOT, 'controller/mu_rot_kmp_axis_angle.npy'))
    poses = np.vstack((pos[:3,:], rot))
    poses[2, :] -= 0.0045 # This is just because the single point task has a wrong origin offset
    forces = np.load(join(ROOT, 'trained_models/mu_force_kmp.npy'))[:3,:]
    # Pull the KMP uncertainties
    pos_uncert = np.load(join(ROOT, 'trained_models/sigma_pos_kmp.npy'))
    pos_uncert = symmetrize(pos_uncert)
    rot_uncert = np.load(join(ROOT, 'trained_models/sigma_rot_kmp.npy'))
    rot_uncert = symmetrize(rot_uncert)
    forces_uncert = np.load(join(ROOT, 'trained_models/sigma_force_kmp.npy'))
    forces_uncert = symmetrize(forces_uncert)
    # Precompute precisions
    if not exists(join(ROOT, 'controller/sigma_pos_kmp_inv.npy')):
        pos_uncert_inv = np.zeros_like(pos_uncert)
        for i in range(pos_uncert.shape[2]):
            pos_uncert_inv[:,:,i] = np.matrix.round(np.linalg.inv(pos_uncert[:,:,i]), 2)
        np.save(join(ROOT, 'controller/sigma_pos_kmp_inv.npy'), pos_uncert_inv)
    else:
        pos_uncert_inv = np.load(join(ROOT, 'controller/sigma_pos_kmp_inv.npy'))
    if not exists(join(ROOT, 'controller/sigma_rot_kmp_inv.npy')):
        rot_uncert_inv = np.zeros_like(rot_uncert)
        for i in range(rot_uncert.shape[2]):
            rot_uncert_inv[:,:,i] = np.matrix.round(np.linalg.inv(rot_uncert[:,:,i]), 2)
        np.save(join(ROOT, 'controller/sigma_rot_kmp_inv.npy'), rot_uncert_inv)
    else:
        rot_uncert_inv = np.load(join(ROOT, 'controller/sigma_rot_kmp_inv.npy'))
    if not exists(join(ROOT, 'controller/sigma_force_kmp_inv.npy')):
        force_uncert_inv = np.zeros_like(forces_uncert)
        for i in range(rot_uncert.shape[2]):
            force_uncert_inv[:,:,i] = np.matrix.round(np.linalg.inv(forces_uncert[:,:,i]), 2)
        np.save(join(ROOT, 'controller/sigma_force_kmp_inv.npy'), force_uncert_inv)
    else:
        rot_uncert_inv = np.load(join(ROOT, 'controller/sigma_rot_kmp_inv.npy'))
    # Compute the gains for the position controller using LQR
    if not exists(join(ROOT, 'controller/KP_pos_gains.npy')):
        A = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])
        B = np.block([[np.zeros((3,3))],[np.eye(3)]])
        gain_magnitude_penalty = 20000
        R = np.eye(3)*gain_magnitude_penalty
        # Gain computation
        KP_pos = []
        KD_pos = []
        for i in range(pos_uncert_inv.shape[2]):
            Q = pos_uncert_inv[:,:,i]
            Q = (Q + Q.T) / 2
            K, S, E = lqr(A, B, Q, R)
            KP_pos.append(np.diag((K[0,0],K[1,1],K[2,2])))
            KD_pos.append(np.diag((K[0,3],K[1,4],K[2,5])))
        KP_pos = np.array(KP_pos)
        KD_pos = np.array(KD_pos)
        np.save(join(ROOT, 'controller/KP_pos_gains.npy'), KP_pos)
        np.save(join(ROOT, 'controller/KD_pos_gains.npy'), KD_pos)
    else:
        KP_pos = np.load(join(ROOT, 'controller/KP_pos_gains.npy'))
        KD_pos = np.load(join(ROOT, 'controller/KD_pos_gains.npy'))
    # Compute the gains for the orientation controller using LQR
    if not exists(join(ROOT, 'controller/KP_rot_gains.npy')):
        A = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])
        B = np.block([[np.zeros((3,3))],[np.eye(3)]])
        gain_magnitude_penalty = 10000
        R = np.eye(3)*gain_magnitude_penalty
        # Gain computation
        KP_rot = []
        KD_rot = []
        for i in range(rot_uncert_inv.shape[2]):
            Q = rot_uncert_inv[:,:,i]
            Q = (Q + Q.T) / 2
            K, S, E = lqr(A, B, Q, R)
            KP_rot.append(np.diag((K[0,0],K[1,1],K[2,2])))
            KD_rot.append(np.diag((K[0,3],K[1,4],K[2,5])))
        KP_rot = np.array(KP_rot)
        KD_rot = np.array(KD_rot)
        np.save(join(ROOT, 'controller/KP_rot_gains.npy'), KP_rot)
        np.save(join(ROOT, 'controller/KD_rot_gains.npy'), KD_rot)
    else:
        KP_rot = np.load(join(ROOT, 'controller/KP_rot_gains.npy'))
        KD_rot = np.load(join(ROOT, 'controller/KD_rot_gains.npy'))# Compute the gains for the orientation controller using LQR
    if not exists(join(ROOT, 'controller/KP_force_gains.npy')):
        A = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])
        B = np.block([[np.zeros((3,3))],[np.eye(3)]])
        gain_magnitude_penalty = 10000
        R = np.eye(3)*gain_magnitude_penalty
        # Gain computation
        KP_force = []
        KI_force = []
        for i in range(force_uncert_inv.shape[2]):
            Q = force_uncert_inv[:,:,i]
            Q = (Q + Q.T) / 2
            K, S, E = lqr(A, B, Q, R)
            KP_force.append(np.diag((K[0,0],K[1,1],K[2,2])))
            KI_force.append(np.diag((K[0,3],K[1,4],K[2,5])))
        KP_force = np.array(KP_force)
        KI_force = np.array(KI_force)
        np.save(join(ROOT, 'controller/KP_force_gains.npy'), KP_force)
        np.save(join(ROOT, 'controller/KI_force_gains.npy'), KI_force)
    else:
        KP_force = np.load(join(ROOT, 'controller/KP_force_gains.npy'))
        KI_force = np.load(join(ROOT, 'controller/KI_force_gains.npy'))
    #These look like good values
    KD_pos = np.eye(3)*0.5
    KP_pos = np.eye(3)*50
    KD_rot = np.eye(3)*0.1
    KP_rot = np.eye(3)*5
    KP_force = np.eye(3)*0.1
    KI_force = np.eye(3)*0.001
    # Setup the force sensor
    hex12 = HEX12(callback=hex12_callback)
    # Go home
    rtde_c.moveJ([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    # Go to the start of the trajectory
    pose = poses[:,0]
    rtde_c.moveJ_IK(pose=pose, speed = 0.5)
    # Execute the trajectory
    dt = 1/500
    for i in range(poses.shape[-1]):
        t_start = rtde_c.initPeriod()
        # Get the state of the robot
        current_pose = rtde_r.getActualTCPPose()
        current_speed = rtde_r.getActualTCPSpeed()
        # Compute the force errors
        force_error = (forces[:,i] - current_force).reshape(-1,1)
        i_force_error = i_force_error + force_error * dt
        # Output of the outer force loop
        dx_force = (KP_force@force_error + KI_force@i_force_error).flatten()
        # Compute the pose error
        pos_error = (dx_force + poses[:3,i] - current_pose[:3]).reshape(-1,1)
        rot_error = (poses[3:,i] - current_pose[3:]).reshape(-1,1)
        # Compute the velocity command
        dxe1 = (KD_pos@(KP_pos@pos_error - np.array(current_speed[:3]).reshape(-1,1))).flatten()
        dxe2 = (KD_rot@(KP_rot@rot_error - np.array(current_speed[3:]).reshape(-1,1))).flatten()
        dxe = np.concatenate((dxe1, dxe2))
        # Send the commands
        rtde_c.speedL(dxe)
        rtde_c.waitPeriod(t_start)
    rtde_c.speedStop()
    # Go home
    rtde_c.moveJ([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    rtde_c.stopScript()

if __name__ == '__main__':
    main()