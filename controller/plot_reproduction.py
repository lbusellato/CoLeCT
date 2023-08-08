import copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from os.path import abspath, dirname, join

ROOT = dirname(abspath(__file__))
# Get the reproduced task data
actual_force = np.load(join(ROOT, "recorded_forces.npy"))
actual_pose = np.load(join(ROOT, "recorded_poses.npy"))
# Get the target task data
target_pos = np.load(join(ROOT, 'mu_pos_kmp.npy'))[:3,:]
target_pos[2,:]-= 0.006 # This is just because the single point task has a wrong origin offset
target_rot = np.load(join(ROOT, 'kmp_rot_vectors.npy'))
target_rot = target_rot[:3,:]
target_pose = np.vstack((target_pos, target_rot)).T
target_force = np.load(join(ROOT, 'mu_force_kmp.npy'))[:3,:].T

target_force[:,0] *= -1
target_force[:,1] *= -1
# Plot everything
red_patch = mpatches.Patch(color='red', label='Target')
blue_patch = mpatches.Patch(color='blue', label='Actual')
plt.ion()
# Plot poses and speeds
fig, ax = plt.subplots(1,6,figsize=(16,8))
labels = ["x [m]", "y [m]", "z [m]", "rx [rad]", "ry [rad]", "rz [rad]"]
for i in range(6):
    ax[i].plot(target_pose[:,i], color='red')
    ax[i].plot(actual_pose[:,i], color='blue')
    ax[i].set_ylabel(labels[i])
    ax[i].grid(True)
fig.suptitle("Reproduced vs target poses and speeds")
fig.tight_layout()
fig.legend(handles=[red_patch, blue_patch])
plt.show()
# Plot forces
fig, ax = plt.subplots(1,3,figsize=(16,8))
labels = ["Fx [N]", "Fy [N]", "Fz [N]"]
for i in range(3):
    ax[i].plot(target_force[:,i], color='red')
    ax[i].plot(actual_force[:,i], color='blue')
    ax[i].set_ylabel(labels[i])
    ax[i].set_xlabel("Time [ms]")
    ax[i].grid(True)
fig.suptitle("Reproduced vs target forces")
fig.tight_layout()
fig.legend(handles=[red_patch, blue_patch])
plt.show()
input()