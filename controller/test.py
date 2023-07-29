
import numpy as np
from os.path import abspath, dirname, join
import matplotlib.pyplot as plt


ROOT = dirname(abspath(__file__))
test = np.load(join(ROOT, "recorded_forces.npy"))
test2 = np.load(join(ROOT, "recorded_poses.npy"))
tes3t = np.load(join(ROOT, "recorded_speeds.npy"))
fig, ax = plt.subplots(1,3,figsize=(16,8))
fx = test[:,0]
fy = test[:,1]
fz = test[:,2]
ax[0].plot(fx)
ax[1].plot(fy)
ax[2].plot(fz)
ax[0].grid()
ax[1].grid()
ax[2].grid()
plt.show()