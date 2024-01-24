import csv
import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import dirname, abspath, join

ROOT = dirname(dirname(abspath(__file__)))

actual_data_head = ['actual_TCP_pose_0', 
                    'actual_TCP_pose_1', 
                    'actual_TCP_pose_2', 
                    'actual_TCP_force_0', 
                    'actual_TCP_force_1', 
                    'actual_TCP_force_2',]
target_data_head = ['kmp_target_TCP_pose_0', 
                    'kmp_target_TCP_pose_1', 
                    'kmp_target_TCP_pose_2', 
                    'target_TCP_force_0', 
                    'target_TCP_force_1', 
                    'target_TCP_force_2',]

fig, ax = plt.subplots(2, 3, figsize=(20,6))
t0 = None
files = listdir(join(ROOT, "reproductions"))
files.sort()
for file in files:
    actual_data = []
    target_data = []
    with open(join(ROOT, "reproductions", file), newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=" ")
        rows = [row for row in reader]
        t = np.array([float(row['timestamp']) for row in rows])
        t -= t[0]
        for h in actual_data_head:
            if h in ['actual_TCP_pose_3','actual_TCP_pose_4','actual_TCP_pose_5']:
                actual_data.append(np.array(np.abs([float(row[h]) for row in rows])))
            else:
                actual_data.append(np.array([float(row[h]) for row in rows]))
        for h in target_data_head:
            target_data.append(np.array([float(row[h]) for row in rows]))
            
    axes = ["x [m]",
            "y [m]",
            "z [m]",
            "$F_x$ [N]",
            "$F_y$ [N]",
            "$F_z$ [N]",
            ]

    for i in range(2):
        for j in range(3):
            sign = 1
            # This is just because the target Fz has opposite sign wrt the one measured by the sensor
            if axes[i * 3 + j] in ["$F_z$ [N]"]:
                sign = -1 
            if i == 0 and j == 0:
                ax[i,j].plot(t, actual_data[i * 3 + j], label = "actual")
                ax[i,j].plot(t, target_data[i * 3 + j], label = "target")
            else:
                ax[i,j].plot(t, sign * actual_data[i * 3 + j])
                ax[i,j].plot(t, target_data[i * 3 + j])
            ax[i,j].set_ylabel(axes[i * 3 + j])
            if i == 1:
                ax[i,j].set_xlabel("Time [s]")
            min_y = round(min(min(sign * actual_data[i * 3 + j]), min(target_data[i * 3 + j])), 3) - 0.2
            max_y = round(max(max(sign * actual_data[i * 3 + j]), max(target_data[i * 3 + j])), 3) + 0.2
            ax[i,j].set_ylim(min_y, max_y)
            ax[i,j].grid()

handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

fig.suptitle("Sliding task - Reproduction")
fig.tight_layout()
plt.show()