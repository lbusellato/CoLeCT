import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import joblib
import time

from itertools import product
from os.path import dirname, abspath, join
from colect.dataset import load_datasets, as_array, from_array
from colect.datatypes import Quaternion
from colect.mixture import GaussianMixtureModel
from colect.kmp import KMP
from colect.utils import linear_traj_w_midpoint_stop, linear_traj

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)

ROOT = dirname(dirname(dirname(abspath(__file__))))

# One every 'subsample' data points will be kept for each demonstration
subsample = 10

def main():

    # Load the demonstrations
    datasets = load_datasets("demonstrations/experiment4")
    datasets = datasets[:5]
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]

    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    dt = 0.001
    demo_duration = dt * (N + 1)  # Duration of each demonstration


    # Input for GMR
    dataset_arrs = [as_array(dataset) for dataset in datasets]
    poses = np.stack((dataset_arrs), axis=2)
    x_gmr = np.mean(poses, axis=2)[:, [2,3,4]].T

    # Demonstrations
    X_force = extract_input_data(datasets)

    # New trajectory for KMP
    start_pose = np.array([-0.365, 0.290, 0.02])
    end_pose = np.array([-0.465, 0.290, 0.02])
    x_kmp, _ = linear_traj_w_midpoint_stop(start_pose, end_pose, n_points=x_gmr.shape[1], n_stop=20)
    x_kmp = x_kmp.T
    x_kmp2, midpoint_idx = linear_traj_w_midpoint_stop(start_pose, end_pose, n_points=250, n_stop=85)
    x_kmp2 = x_kmp2.T

    gmm = GaussianMixtureModel(n_components=10, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x_gmr)

    kmp = KMP(l=1e-4, alpha=5e4, sigma_f=1e4, time_driven_kernel=False)
    kmp.fit(x_kmp, mu_force, sigma_force)

    width = 5
    waypoint_pos = x_kmp2[:, midpoint_idx[0] - width:midpoint_idx[-1] + width]
    waypoint_forces = np.zeros_like(waypoint_pos)
    waypoint_forces[2, :] = 0 * np.ones_like(waypoint_forces[2, :])
    waypoint_sigmas = np.array([1e-9*np.eye(3) for _ in range(waypoint_pos.shape[1])]).transpose(1, 2, 0)

    kmp.set_waypoint(waypoint_pos, waypoint_forces, waypoint_sigmas)

    mu_kmp, sigma_kmp = kmp.predict(x_kmp2)


    fig_vs, ax = plt.subplots(2, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)

    t_gmr = dt * np.arange(0,x_gmr.shape[1])
    t_kmp = dt * (x_gmr.shape[1] / x_kmp2.shape[1]) * np.arange(0,x_kmp2.shape[1])
    for i in range(3):
        ax[0,i].plot(t_gmr, x_gmr[i, :],color="red")
        ax[0,i].plot(t_kmp, x_kmp2[i, :],color="green")
        ax[1,i].plot(t_gmr, mu_force[i, :],color="red")
        ax[1,i].plot(t_kmp, mu_kmp[i, :],color="green")
        ax[1,i].fill_between(x=t_kmp, y1=mu_kmp[i, :]+np.sqrt(sigma_kmp[i, i, :]), y2=mu_kmp[i, :]-np.sqrt(sigma_kmp[i, i, :]),color="green",alpha=0.35)       

    fig_vs.suptitle("Experiment 4")
    fig_vs.tight_layout()

    plt.show()


def extract_input_data(datasets: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Creates the input array for training the GMMs

    Parameters
    ----------
    datasets : np.ndarray
        The demonstration dataset(s).
    dt : float, default = 0.1
        Timestep for the creation of the input vector (timeseries) and for the computation of the derivatives of the output.

    Returns
    -------
    np.ndarray
        An array of shape (n_samples, n_features) containing the input for training the GMMs.
    """

    # Extract the shape of each demonstration
    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration
    
    pos = np.vstack([p.position for dataset in datasets for p in dataset]).T
    force = np.vstack([p.force for dataset in datasets for p in dataset]).T

    # Add the derivatives to the dataset
    X = np.vstack((pos, force))

    # The shape of the dataset should be (n_samples, n_features)
    return X.T


def plot_demo(
    ax: plt.Axes, demonstration: np.ndarray, duration: float, dt: float = 0.1
):
    """Plot the position, orientation (as Euclidean projection of quaternions) and force data contained in a demonstration.

    Parameters
    ----------
    ax : plt.Axes
        The plot axes on which to plot the data.
    demonstration : np.ndarray
        The array containing the demonstration data.
    duration : float
        Total trajectory duration, used to correctly display time.
    duration : float, default = 0.1
        Trajectory time step, used to correctly display time.
    """

    time = [p.time for p in demonstration] 
    time = 0.001 * np.arange(1, len(time) + 1)
    # Recover data
    x = [p.x for p in demonstration]
    y = [p.y for p in demonstration]
    z = [p.z for p in demonstration]
    fx = [p.fx for p in demonstration]
    fy = [p.fy for p in demonstration]
    fz = [p.fz for p in demonstration]
    data = [x, y, z, fx, fy, fz]
    y_labels = [
        "x [m]",
        "y [m]",
        "z [m]",
        "$F_x$ [N]",
        "$F_y$ [N]",
        "$F_z$ [N]",
    ]
    # Plot everything
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(time, data[i * 3 + j], linewidth=0.6, color="grey")
            ax[i, j].set_ylabel(y_labels[i * 3 + j])
            ax[i, j].grid(True)
            if i == 2:
                ax[i, j].set_xlabel("Time [s]")


if __name__ == "__main__":
    main()