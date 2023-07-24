import rtde_control
import rtde_receive
import numpy as np
import time
from control import lqr
from os.path import join, dirname, abspath, exists

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

rtde_c = rtde_control.RTDEControlInterface("192.168.1.11")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")

# Pull the KMP trajectories
pos = np.load(join(ROOT, 'trained_models/mu_pos_kmp.npy'))
if not exists(join(ROOT, 'trained_models/mu_rot_kmp_axis_angle.npy')):
    rot = np.load(join(ROOT, 'trained_models/mu_rot_kmp_quats.npy'))
    rot = to_axis_angle(rot)
    np.save(join(ROOT, 'trained_models/mu_rot_kmp_axis_angle.npy'), rot)
else:
    rot = np.load(join(ROOT, 'trained_models/mu_rot_kmp_axis_angle.npy'))
poses = np.vstack((pos[:3,:], rot))
# Pull the KMP uncertainties
pos_uncert = np.load(join(ROOT, 'trained_models/sigma_pos_kmp.npy'))
pos_uncert = symmetrize(pos_uncert)
rot_uncert = np.load(join(ROOT, 'trained_models/sigma_rot_kmp.npy'))
rot_uncert = symmetrize(rot_uncert)
# Precompute precisions
if not exists(join(ROOT, 'trained_models/sigma_pos_kmp_inv.npy')):
    pos_uncert_inv = np.zeros_like(pos_uncert)
    for i in range(pos_uncert.shape[2]):
        pos_uncert_inv[:,:,i] = np.matrix.round(np.linalg.inv(pos_uncert[:,:,i]), 2)
    np.save(join(ROOT, 'trained_models/sigma_pos_kmp_inv.npy'), pos_uncert_inv)
else:
    pos_uncert_inv = np.load(join(ROOT, 'trained_models/sigma_pos_kmp_inv.npy'))
if not exists(join(ROOT, 'trained_models/sigma_rot_kmp_inv.npy')):
    rot_uncert_inv = np.zeros_like(rot_uncert)
    for i in range(rot_uncert.shape[2]):
        rot_uncert_inv[:,:,i] = np.matrix.round(np.linalg.inv(rot_uncert[:,:,i]), 2)
    np.save(join(ROOT, 'trained_models/sigma_rot_kmp_inv.npy'), rot_uncert_inv)
else:
    rot_uncert_inv = np.load(join(ROOT, 'trained_models/sigma_rot_kmp_inv.npy'))
# Compute the gains for the position controller using LQR
if not exists(join(ROOT, 'trained_models/KP_pos_gains.npy')):
    A = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])
    B = np.block([[np.zeros((3,3))],[np.eye(3)]])
    gain_magnitude_penalty = 1000
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
    np.save(join(ROOT, 'trained_models/KP_pos_gains.npy'), KP_pos)
    np.save(join(ROOT, 'trained_models/KD_pos_gains.npy'), KD_pos)
else:
    KP_pos = np.load(join(ROOT, 'trained_models/KP_pos_gains.npy'))
    KD_pos = np.load(join(ROOT, 'trained_models/KD_pos_gains.npy'))
# Compute the gains for the orientation controller using LQR
if not exists(join(ROOT, 'trained_models/KP_rot_gains.npy')):
    A = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])
    B = np.block([[np.zeros((3,3))],[np.eye(3)]])
    gain_magnitude_penalty = 1000
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
    np.save(join(ROOT, 'trained_models/KP_rot_gains.npy'), KP_rot)
    np.save(join(ROOT, 'trained_models/KD_rot_gains.npy'), KD_rot)
else:
    KP_rot = np.load(join(ROOT, 'trained_models/KP_rot_gains.npy'))
    KD_rot = np.load(join(ROOT, 'trained_models/KD_rot_gains.npy'))

# Go home
rtde_c.moveJ([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
# Go to the start of the trajectory
pose = poses[:,0]
rtde_c.moveJ_IK(pose=pose, speed = 0.5)
# Execute the trajectory
for i in range(poses.shape[-1]):
    t_start = rtde_c.initPeriod()
    # Get the state of the robot
    current_pose = rtde_r.getActualTCPPose()
    current_speed = rtde_r.getActualTCPSpeed()
    # Compute the pose error
    xf1 = (poses[:3,i] - current_pose[:3]).reshape(-1,1)
    xf2 = (poses[3:,i] - current_pose[3:]).reshape(-1,1)
    # Compute the velocity command
    dxe1 = (KD_pos[i,:,:]@(KP_pos[i,:,:]@xf1 - np.array(current_speed[:3]).reshape(-1,1))).flatten()
    dxe2 = (KD_rot[i,:,:]@(KP_rot[i,:,:]@xf2 - np.array(current_speed[3:]).reshape(-1,1))).flatten()
    dxe = np.concatenate((dxe1, [0.0,0.0,0.0]))#dxe2))
    # Send the commands
    rtde_c.speedL(dxe)
    rtde_c.waitPeriod(t_start)
# Go home
rtde_c.moveJ([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
rtde_c.speedStop()
rtde_c.stopScript()
