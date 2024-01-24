import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import os
import PIL.Image

from colect.dataset import load_datasets, as_array
from glob import glob
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from os.path import dirname, abspath, join

ROOT = dirname(dirname(dirname(abspath(__file__))))


def plot_demo(ax, demonstration, linewidth=1.0, color='blue'):
    # Plot all the data in a demonstration
    time = [0.001*i for i in range(len(demonstration))]
    x = [p.x for p in demonstration]
    y = [p.y for p in demonstration]
    z = [p.z for p in demonstration]
    qx = [p.rot.as_array()[1] for p in demonstration]
    qy = [p.rot.as_array()[2] for p in demonstration]
    qz = [p.rot.as_array()[3] for p in demonstration]
    fx = [p.fx for p in demonstration]
    fy = [p.fy for p in demonstration]
    fz = [p.fz for p in demonstration]
    tx = [p.mx for p in demonstration]
    ty = [p.my for p in demonstration]
    tz = [p.mz for p in demonstration]
    data = np.array([time, x, y, z, qx, qy, qz, fx, fy, fz, tx, ty, tz])
    y_labels = ['x [m]', 'y [m]', 'z [m]',
                '$q_x$', '$q_y$', '$q_z$',
                '$F_x$ [N]', '$F_y$ [N]', '$F_z$ [N]',
                '$M_x$ - [Nmm]', '$M_y$ - [Nmm]', '$M_z$ - [Nmm]']
    for i in range(4):
        for j in range(3):
            ax[i, j].plot(data[0], data[i*3 + j + 1],
                          linewidth=linewidth, color=color)
            ax[i, j].set_ylabel(y_labels[i*3 + j])
            ax[i, j].grid(True)
            if i == 3:
                ax[i, j].set_xlabel('Time [s]')

def main():
    """This script parses all images related to the demos, saving the ones that are actually seen
    during the demo in a subfolder.
    """

    for subject in range(1,6):
        path = 'demonstrations/experiment5/subject' + str(subject)
        # Load the processed datasets
        datasets = load_datasets(path)

        # Get each image's timestamp
        path = 'us_probe_recordings/'
        images = [f for f in os.listdir(join(ROOT, path)) if os.path.isfile(join(ROOT, path, f))]
        images.sort()
        
        for i, dataset in enumerate(datasets):
            dataset_timestamps = np.array([p.timestamp for p in dataset])

            os.makedirs(join(ROOT, "subject" + str(subject), f"dataset{i:02d}"), exist_ok=True)
            for img_name in images:
                img_timestamp = float(os.path.basename(img_name[:-4]))

                # Check if the image is valid
                absolute_diff = np.abs(dataset_timestamps - img_timestamp)
                min_diff = np.min(absolute_diff)
                if min_diff < 1:
                    img = PIL.Image.open(join(ROOT, path, img_name))
                    img.save(join(ROOT, "subject" + str(subject), f"dataset{i:02d}", img_name))

if __name__ == '__main__':
    main()
