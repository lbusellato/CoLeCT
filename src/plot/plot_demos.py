import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from fastdtw import fastdtw
from os.path import abspath, dirname

# Collection of interesting plots


def signal_sync_test():
    # Show that the force readings and the pose readings are indeed synchronized
    recording_path = os.path.join(
        dirname(dirname(abspath(__file__))), 'plot/demonstrations/signal_sync_test')
    file = os.path.join(recording_path, 'signal_sync_test.csv')
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
            float(col['pos_y']) - y0,
            (float(col['force_z']) - f0)/1000,]
        demo.append(point)
    demo = np.array(demo)
    demo = demo / np.linalg.norm(demo)
    ax.plot(demo[:, 0], label='Position')
    ax.plot(demo[:, 1], label='Force')
    ax.set_title('Position-force synchronization test')
    ax.set_xlabel('Time [ms]')
    ax.legend()
    ax.grid()
    plots_path = os.path.join(
        dirname(dirname(abspath(__file__))), 'plot/plots')
    plt.savefig(os.path.join(plots_path, 'signal_sync_test.png'))
    plt.show()

def dtw():
    # Using DTW (Dynamic Time Warping) to achieve temporal alignment of demonstrations
    demo_path = os.path.join(
        dirname(dirname(abspath(__file__))), 'plot/demonstrations/dtw')
    res = [path for path in os.listdir(demo_path) if os.path.isfile(os.path.join(demo_path, path))]
    ax = plt.figure().add_subplot()
    demonstrations = []
    # Recover the demonstrations from the .csv files
    for i, path in enumerate(res):
        filename = open(os.path.join(demo_path, path), 'r')
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
            distance, path = fastdtw(demonstrations[reference_demo][:], demo[:], dist=2)
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
    for i,demo in enumerate(aligned_demos):
        if i != reference_demo:
            ax.plot(demo[:], label=str(i)+'-aligned')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time [ms]')
    ax.set_title('Signal alignment with DTW')
    plots_path = os.path.join(
        dirname(dirname(abspath(__file__))), 'plot/plots')
    plt.savefig(os.path.join(plots_path, 'dtw.png'))
    plt.show()


def main(demo_number):
    if demo_number == 1:
        signal_sync_test()
    elif demo_number == 2:
        dtw()
    else:
        pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[-1]))