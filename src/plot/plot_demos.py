import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

from os import listdir
from os.path import abspath, dirname, isfile, join
from fastdtw import fastdtw


# Collection of interesting plots

ROOT = dirname(dirname(dirname(abspath(__file__))))


def demo_to_array(dataset):
    ret = []
    for point in dataset:
        ret.append([point.timestamp, point.x, point.y, point.z, point.rot.as_array()[0], point.rot.as_array()[
                   1], point.rot.as_array()[2], point.rot.as_array()[3], point.rot_eucl[0], point.rot_eucl[1], point.rot_eucl[2], point.fx, point.fy, point.fz, point.mx, point.my, point.mz])
    return np.array(ret)


def plot_demo(ax, demonstration, linestyle='solid', label=''):
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
    data = [x, y, z, qx, qy, qz, fx, fy, fz, tx, ty, tz]
    y_labels = ['x [m]', 'y [m]', 'z [m]',
                'qx', 'qy', 'qz',
                'Fx [N]', 'Fy [N]', 'Fz [N]',
                'Mx - [Nm]', 'My - [Nm]', 'Mz - [Nm]']
    for i in range(4):
        for j in range(3):
            ax[i, j].plot(time, data[i*3 + j],
                          linestyle=linestyle, label=label)
            ax[i, j].set_ylabel(y_labels[i*3 + j])
            ax[i, j].grid()
            if i == 3:
                ax[i, j].set_xlabel('Time [s]')


def signal_sync_test():
    # Show that the force readings and the pose readings are indeed synchronized
    demo_path = join(ROOT, 'demonstrations/signal_sync_test')
    file = join(demo_path, 'signal_sync_test.csv')
    ax = plt.figure().add_subplot()
    # Recover the demonstration from the .csv file
    filename = open(file, 'r')
    file = csv.DictReader(filename)
    demo = []
    y0 = 0
    f0 = 0
    for col in file:
        if y0 == 0:
            y0 = float(col['pos_y'])
            f0 = float(col['force_z'])
        point = [
            float(col['pos_y']) - y0,  # Match the sign of the force
            (float(col['force_z']) - f0)/1000,]  # Scale just for visualization
        demo.append(point)
    demo = np.array(demo)
    demo = demo / np.linalg.norm(demo)
    ax.plot(demo[:, 0], label='Position')
    ax.plot(demo[:, 1], label='Force')
    ax.set_title('Position-force synchronization test')
    ax.set_xlabel('Time [ms]')
    ax.legend()
    ax.grid()
    plots_path = join(ROOT, 'media')
    plt.savefig(join(plots_path, 'signal_sync_test.png'))
    plt.show()


def dtw():
    # Using DTW (Dynamic Time Warping) to achieve temporal alignment of demonstrations
    demo_path = join(ROOT, 'demonstrations/dtw')
    res = [path for path in listdir(
        demo_path) if isfile(join(demo_path, path))]
    ax = plt.figure().add_subplot()
    demonstrations = []
    # Recover the demonstrations from the .csv files
    for i, path in enumerate(res):
        filename = open(join(demo_path, path), 'r')
        file = csv.DictReader(filename)
        demo = []
        for col in file:
            point = [
                float(col['pos_y'])]
            demo.append(point)
        demo = np.array(demo)
        ax.plot(demo, label=i, linestyle='dashed')
        demonstrations.append(demo)
    # Find the velocity-wise mean demonstration
    lens = [len(x) for x in demonstrations]
    mean = np.mean(lens)
    diff = []
    for i, value in enumerate(lens):
        diff.append((abs(value-mean), i))
    _, reference_demo = min(diff)
    # Compute the warping path with DTW
    distances = []
    paths = []
    for i, demo in enumerate(demonstrations):
        if i != reference_demo:
            distance, path = fastdtw(
                demonstrations[reference_demo][:], demo[:], dist=2)
            distances.append(distance)
            paths.append(path)
        else:
            distances.append(0.0)
            paths.append([])
    # Apply the warping path to align the signals temporally
    aligned_demos = [demonstrations[reference_demo]]
    for i, path in enumerate(paths):
        if i != reference_demo:
            aligned_demo = []
            prev_x = -1
            for [x, y] in path:
                if x != prev_x:
                    aligned_demo.append(demonstrations[i][y])
                prev_x = x
            aligned_demos.append(np.array(aligned_demo))
    for i, demo in enumerate(aligned_demos):
        if i != reference_demo:
            ax.plot(demo[:], label=str(i)+'-aligned')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time [ms]')
    ax.set_title('Signal alignment with DTW')
    plots_path = join(ROOT, 'media')
    plt.savefig(join(plots_path, 'dtw.png'))
    plt.show()


def interpolation():
    demo_path = join(ROOT, 'demonstrations/interpolation_test')
    res = [join('demonstrations/interpolation_test', path) for path in listdir(
        demo_path) if 'dataset' in path and isfile(join(demo_path, path))]
    fig, ax = plt.subplots(4, 3)
    # Recover the demonstrations from the .csv files
    for i, path in enumerate(res):
        demo = np.load(join(ROOT, path), allow_pickle=True)
        plot_demo(ax, demo, label=i)
    plt.show()


def main(demo_number):
    if demo_number == 1:
        signal_sync_test()
    elif demo_number == 2:
        dtw()
    elif demo_number == 3:
        interpolation()
    else:
        pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[-1]))
