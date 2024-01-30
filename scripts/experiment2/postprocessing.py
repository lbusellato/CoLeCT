import matplotlib.pyplot as plt
import numpy as np

from os.path import join, dirname, abspath
from colect.dataset import *

ROOT = dirname(dirname(dirname(abspath(__file__))))


def plot_demo(ax, demonstration, linewidth=1.0, color='blue', label=""):
    # Plot all the data in a demonstration
    time = [p.time for p in demonstration]
    x = [p.x for p in demonstration]
    y = [p.y for p in demonstration]
    z = [p.z for p in demonstration]
    qx = [p.rot.as_array()[0] for p in demonstration]
    qy = [p.rot.as_array()[1] for p in demonstration]
    qz = [p.rot.as_array()[2] for p in demonstration]
    qw = [p.rot.as_array()[3] for p in demonstration]
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
                          linewidth=linewidth, label=label)
            ax[i, j].set_ylabel(y_labels[i*3 + j])
            ax[i, j].grid(True)
            if i == 3:
                ax[i, j].set_xlabel('Time [s]')
    for i in range(4):
        for j in range(4):
            if i != 1 and j == 3:
                ax[i, j].axis('off')
    ax[1, 3].plot(data[0], qw, linewidth=linewidth)
    ax[1, 3].set_ylabel('$q_w$')
    ax[1, 3].grid(True)
    ax[1, 3].set_xlabel('Time [s]')


def main():
    # Showcase the dataset postprocessing operations
    # Process the .csv files into .npy files
    path = 'demonstrations/experiment2'
    create_dataset(path, 20)
    # Trim any leading or trailing force-only samples
    trim_datasets(path)
    # Fill in the force-only samples by linearly interpolating the poses
    interpolate_datasets(path)
    # Transform the coordinates to the base robot frame
    to_base_frame(path)
    # Align demos wrt Fz
    align_datasets(path)
    # Load the processed datasets
    processed = load_datasets(path)
    # Plot everything
    plt.ion()
    fig, ax = plt.subplots(4, 4, figsize=(16, 8))
    for i, dataset in enumerate(processed):
        plot_demo(ax, dataset, label=i)
    fig.suptitle('Dataset postprocessing')
    fig.tight_layout()
    fig.legend()
    plt.show(block=False)
    lower_t = input("Lower time cutoff (0 for no cutoff): ")
    upper_t = input("Upper time cutoff (0 for no cutoff): ")
    clip_datasets(path, float(lower_t), float(upper_t))
    flip_fz(path)
    check_quat_signs(path)
    processed = load_datasets(path)
    fz = [p.fz for dataset in processed for p in dataset]
    print(f"Force z mean: {np.mean(fz)}, std: {np.std(fz)}")
    fig, ax = plt.subplots(4, 4, figsize=(16, 8))
    for i, dataset in enumerate(processed):
        plot_demo(ax, dataset, linewidth=0.75)
    fig.suptitle('Dataset postprocessing')
    fig.tight_layout()
    fig.legend()
    plt.show(block=False)
    input()


if __name__ == '__main__':
    main()
