import rtde_control
import rtde_receive
import numpy as np
import time
from control import lqr
from os.path import join, dirname, abspath

ROOT = dirname(dirname(abspath(__file__)))

rtde_c = rtde_control.RTDEControlInterface("172.17.0.2")
rtde_r = rtde_receive.RTDEReceiveInterface("172.17.0.2")

# Pull the KMP trajectories
pos = np.load(join(ROOT, 'trained_models/mu_pos_kmp.npy'))
rot = np.load(join(ROOT, 'trained_models/mu_rot_kmp.npy'))

# Pull the KMP uncertainties
pos_uncert = np.load(join(ROOT, 'trained_models/sigma_pos_kmp.npy'))
rot_uncert = np.load(join(ROOT, 'trained_models/sigma_rot_kmp.npy'))

# Fake velocity
vel = np.gradient(pos)[0]
vel_uncert = np.tile(np.eye(3), pos_uncert.shape[2]).reshape(pos_uncert.shape)

# Precompute precisions

# Zero the FT sensor
#rtde_c.zeroFtSensor()
# Go to the start of the trajectory
print([pos[0,0], pos[1,0], pos[2,0], rot[0,0], rot[1,0], rot[2,0]])
# Execute the trajectory
"""while True:
    t_start = rtde_c.initPeriod()
    rtde_c.waitPeriod(t_start)
rtde_c.stopScript()"""