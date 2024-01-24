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
from colect.utils import linear_traj

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
    datasets = datasets[:4]
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

    # Local frames
    A1 = np.eye(6)
    b1 = np.array([*x_gmr[:,0],0,0,0])#np.zeros(6)
    A2 = np.eye(6)
    b2 = np.array([*x_gmr[:,-1],0,0,0])#np.zeros(6)
    An = [A1, A2]
    bn = [b1, b2]

    # Demonstrations
    X_force = extract_input_data(datasets)

    # New trajectory for KMP
    start_pose = np.array([-0.465, 0.290, 0.05])
    end_pose = np.array([-0.365, 0.290, 0.05])
    x_kmp = linear_traj(start_pose, end_pose, n_points=N)[:, :3].T

    # Project demonstrations and GMR input into local frames
    X_force_p = [np.array([A @ (row - b) for row in zip(*X_force.T)]) for A, b in zip(An, bn)]
    x_gmr_p = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_gmr)]).T for A, b in zip(An, bn)]

    # Extract local reference databases
    mu_gmr_p = []
    sigma_gmr_p = []
    for X, x in zip(X_force_p, x_gmr_p):
        gmm = GaussianMixtureModel(n_components=10, n_demos=H)
        gmm.fit(X)
        mu_force, sigma_force = gmm.predict(x)
        mu_gmr_p.append(mu_force)
        sigma_gmr_p.append(sigma_force)

    # Project the new input into the local frames
    x_kmp_p = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_kmp)]).T for A, b in zip(An, bn)]    

    kmp = KMP(l=1e-5, alpha=1e4, sigma_f=1e5, time_driven_kernel=False)
    mu_kmp_p = []
    sigma_kmp_p = []
    for x_train, x_test, mu, sigma in zip(x_kmp_p, x_kmp_p, mu_gmr_p, sigma_gmr_p):
        kmp.fit(x_train, mu, sigma)
        mu_kmp, sigma_kmp = kmp.predict(x_test)
        mu_kmp_p.append(mu_kmp)
        sigma_kmp_p.append(sigma_kmp)

    mu_force = []
    for i in range(x_kmp.shape[1]):
        sigma_p_inv = np.array([np.linalg.inv(sigma[:,:,i]) for sigma in sigma_kmp_p])
        sigma_p_inv = np.linalg.inv(np.sum(sigma_p_inv, axis=0))
        sigma_p_inv_prod = np.array([np.linalg.inv(sigma[:,:,i])@mu[:,i] for mu, sigma in zip(mu_kmp_p, sigma_kmp_p)])
        sigma_p_inv_prod = np.sum(sigma_p_inv, axis=0)
        mu_force.append(sigma_p_inv @ sigma_p_inv_prod)
    mu_force = np.vstack(mu_force).T

    fig_vs, ax = plt.subplots(2, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)

    t_gmr = dt * np.arange(0,mu_force.shape[1])
    for i in range(3):
        ax[1,i].plot(t_gmr, mu_force[i, :],color="red")

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