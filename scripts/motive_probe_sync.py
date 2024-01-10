import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import os
import PIL.Image

from colect.dataset import load_datasets, as_array
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
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
    # Showcase the dataset postprocessing operations
    # Process the .csv files into .npy files
    path = 'demonstrations/vascular_phantom/'
    # Load the processed datasets
    datasets = load_datasets(path)
    dataset_timestamp = np.array([p.timestamp for p in datasets[0]])
    z_force = [p.fz for p in datasets[0]]
    z_coord = [p.z for p in datasets[0]]
    time = [0.001*i for i in range(len(datasets[0]))]

    # Get each image's timestamp
    img_timestamp = []
    path = 'us_probe_recordings/recording_top_vase'
    images = os.listdir(join(ROOT, path))
    images.sort()
    for img_name in images:
        img = PIL.Image.open(join(ROOT, path, img_name))
        exif_data = img._getexif()
        img_timestamp.append(float(exif_data[37510]))
    img_timestamp = np.array(img_timestamp)
    # Plot the sine wave
    plt.plot(time, z_force)

    # Add labels and title
    plt.xlabel('Time [s]')
    plt.ylabel('Force z [N]')
    plt.title('Z-force vs us image')

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.set_ylabel("Position z [m]")
    plt.plot(time, z_coord,color="red")
    # Enable hover functionality
    cursor = mplcursors.cursor(pickables=[ax],hover=True)

    global last
    last = None

    # Define the hover function
    @cursor.connect("add")
    def on_hover(sel):
        global last
        if last is not None:
            last.remove()

        x_value = sel.target[0]
        y_value = sel.target[1]

        absolute_diff = np.abs(time - x_value)
        closest_index = np.argmin(absolute_diff)
        point_timestamp = dataset_timestamp[closest_index]
        
        absolute_diff = np.abs(img_timestamp - point_timestamp)
        img_index = np.argmin(absolute_diff)

        arr_img = plt.imread(f"./us_probe_recordings/recording_top_vase/{img_index:05d}.png")

        imagebox = OffsetImage(arr_img, zoom=0.3)
        imagebox.image.axes = ax

        xy = (x_value,y_value)
        last = AnnotationBbox(imagebox, xy,
                            xybox=(120., -50.),
                            boxcoords="offset points",
                            pad=0.5,
                            arrowprops=dict(arrowstyle="->")
                            )

        ax.add_artist(last)
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
