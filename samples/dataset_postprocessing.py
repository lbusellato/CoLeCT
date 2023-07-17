import matplotlib.pyplot as plt
import numpy as np

from os.path import join, dirname, abspath
from src.dataset import create_dataset, trim_datasets, align_datasets, interpolate_datasets, load_datasets

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
    # Get rid of empty pose rows
    data = np.array([row for row in data.T if not all(element == 0 for element in row[1:4])]).T
    y_labels = ['x [m]', 'y [m]', 'z [m]',
                '$q_x$', '$q_y$', '$q_z$',
                '$F_x$ [N]', '$F_y$ [N]', '$F_z$ [N]',
                '$M_x$ - [Nmm]', '$M_y$ - [Nmm]', '$M_z$ - [Nmm]']
    for i in range(4):
        for j in range(3):
            ax[i, j].plot(data[0], data[i*3 + j + 1], linewidth=linewidth, color=color)
            ax[i, j].set_ylabel(y_labels[i*3 + j])
            ax[i, j].grid(True)
            if i == 3:
                ax[i, j].set_xlabel('Time [s]')

def main():
    # Showcase the dataset postprocessing operations
    # Process the .csv files into .npy files
    path = 'demonstrations/single_point_task'
    regex = r'single_point_task(\d{2})\.csv'
    create_dataset(path, demonstration_regex=regex)
    # Trim any leading or trailing force-only samples
    trim_datasets(path)
    # Fill in the force-only samples by linearly interpolating the poses
    interpolate_datasets(path)
    # Align temporally the datasets with Soft-DTW
    align_datasets(path)
    # Load the processed datasets
    processed = load_datasets(path)
    # Plot everything
    fig, ax = plt.subplots(4, 3, figsize=(16, 8))
    for dataset in processed:
        plot_demo(ax, dataset, color='blue')
    fig.suptitle('Dataset postprocessing')
    fig.tight_layout()
    plots_path = join(ROOT, 'media/single_point_task.png')
    plt.savefig(plots_path)
    plt.show()


if __name__ == '__main__':
    main()
