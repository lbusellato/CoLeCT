import csv
import matplotlib.pyplot as plt
import numpy as np

from os.path import join, abspath, dirname

ROOT = dirname(dirname(abspath(__file__)))


def main():
    # Show that the force readings and the pose readings are indeed synchronized
    file_path = join(ROOT, 'demonstrations/signal_sync_test/signal_sync_test.csv')
    # Recover the demonstration from the .csv file
    filename = open(file_path, 'r')
    file = csv.DictReader(filename)
    # Scale the force for better visualization
    demo = np.array([[float(col['pos_y']), float(col['force_z'])] for col in file])
    # Plot everything
    ax = plt.figure().add_subplot()
    ax.plot(demo[:, 0], color='blue')
    tmp = ax.twinx()
    tmp.plot(demo[:, 1], color='red')
    ax.set_title('Position-force synchronization')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('y [m]', color='blue')
    tmp.set_ylabel('$F_y$ [N]', color='red')
    ax.grid()
    plots_path = join(ROOT, 'media/signal_sync_test.png')
    plt.savefig(plots_path)
    plt.show()


if __name__ == '__main__':
    main()
