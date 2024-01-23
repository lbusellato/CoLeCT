import numpy as np
import os
import PIL.Image

from colect.dataset import load_reproduction
from glob import glob
from os.path import dirname, abspath, join

ROOT = dirname(dirname(abspath(__file__)))


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

    path = 'reproductions'
    
    datasets = load_reproduction(path)
    dataset_timestamps = [dataset[:,-1] for dataset in datasets]

    # Get each image's timestamp
    path = 'us_probe_recordings'
    subdirs = glob(join(ROOT, path) + "/*/", recursive = True)
    subdirs.sort()
    
    for i, subdir in enumerate(subdirs):
        img_timestamp = []
        os.makedirs(join(subdir, f"dataset{i:02d}"), exist_ok=True)
        images = os.listdir(subdir)
        images.sort()

        img_timestamp = np.array([float(os.path.basename(img[:-4])) for img in images if img.endswith('.png')])

        # Compute which images are valid
        indexes = []
        for t in dataset_timestamps[i]:
            absolute_diff = np.abs(img_timestamp - t)
            indexes.append(np.argmin(absolute_diff))
        res = list(set(indexes))
        for r in res:
            img = PIL.Image.open(join(subdir, images[r]))
            img.save(join(subdir, f"dataset{i:02d}", images[r]))

if __name__ == '__main__':
    main()
