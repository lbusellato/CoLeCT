import numpy as np
from os.path import dirname, abspath, join
import matplotlib.pyplot as plt

ROOT = dirname(dirname(abspath(__file__)))

recorded_poses = np.load(join(ROOT, "recorded_poses.npy"))
target_poses = np.load(join(ROOT, "trained_models/traj.npy"))
dt = 1 / 500 * 5
t_r = dt * np.arange(1, recorded_poses.shape[0] + 1)
t_t = dt * np.arange(1, target_poses.shape[0] + 1)

fig, ax = plt.subplots(1, 3, figsize=(10,6))
ax[0].plot(t_r, recorded_poses[:,0], label="recorded")
ax[1].plot(t_r, recorded_poses[:,1])
ax[2].plot(t_r, recorded_poses[:,2])
ax[0].plot(t_t, target_poses[:,0], label="target")
ax[1].plot(t_t, target_poses[:,1])
ax[2].plot(t_t, target_poses[:,2])
ax[0].legend()

for i in range(3):
    ax[i].set_ylim(min(min(recorded_poses[:,i]), min(target_poses[:,i])) - 0.1,
                   max(max(recorded_poses[:,i]), max(target_poses[:,i])) + 0.1)

plt.show()