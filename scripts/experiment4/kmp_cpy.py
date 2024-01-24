import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import joblib
import time

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

    # Use the average pose as input for GMR prediction
    dataset_arrs = [as_array(dataset) for dataset in datasets]
    poses = np.stack((dataset_arrs), axis=2)
    x_gmr_train = np.mean(poses, axis=2)[:, [2,3,4]].T
    
    # Prepare data for GMM/GMR
    X_force = extract_input_data(datasets)

    # Generate the trajectory
    N_test = N // 2
    start_pose = np.array([-0.365, 0.290, 0.05])
    end_pose = np.array([-0.415, 0.290, 0.05])
    x_kmp_test1 = linear_traj(start_pose, end_pose, n_points=N_test)[:, :3].T

    start_pose2 = np.array([-0.415, 0.290, 0.05])
    end_pose2 = np.array([-0.465, 0.290, 0.05])
    x_kmp_test2 = linear_traj(start_pose2, end_pose2, n_points=N_test)[:, :3].T

    x_kmp_train = linear_traj(start_pose, end_pose2, n_points=N)[:, :3].T

    x_kmp_test = x_kmp_test1
    N_test = x_kmp_test.shape[1]

    #np.save(join(ROOT, "trained_models", "experiment4_traj.npy"), x_kmp_test)

    # An are the rotation matrices for the start and end frame
    An = np.array([np.linalg.inv(np.eye(6)), np.linalg.inv(np.eye(6))])
    # bn are the translation vectors for the start and end frame
    bn = np.array([[*start_pose, 0, 0, 0], [*end_pose, 0, 0, 0], [*start_pose2, 0, 0, 0], [*end_pose2, 0, 0, 0]])
    X_force = extract_input_data(datasets)
    # X_p are the local demonstration databases
    X_p = [np.array([A @ (row - b) for row in zip(*X_force.T)]) for A, b in zip(An, bn)]
    
    # Project the reference trajectory to the local dbs
    x_gmr_p = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_gmr_train)]).T for A, b in zip(An, bn)]
    
    # Extract the local reference databases
    mu_gmr_p = []
    sigma_gmr_p = []
    for X, x_gmr in zip(X_p, x_gmr_p):
        gmm = GaussianMixtureModel(n_components=10, n_demos=H)
        gmm.fit(X)
        mu_force, sigma_force = gmm.predict(x_gmr)
        mu_gmr_p.append(mu_force)
        sigma_gmr_p.append(sigma_force)

    # Project the input into the local frames
    x_kmp_p_train = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_kmp_train)]).T for A, b in zip(An, bn)]
    x_kmp_p_test = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_kmp_test)]).T for A, b in zip(An, bn)]

    # Run KMP on both reference databases
    mu_kmp_p_test = []
    sigma_kmp_p_test = []
    kmp = KMP(l=1e-5, alpha=1e4, sigma_f=1e5, time_driven_kernel=False)
    for x_train, x_test, mu_gmr, sigma_gmr in zip(x_kmp_p_train, x_kmp_p_test, mu_gmr_p, sigma_gmr_p):
        kmp.fit(x_train, mu_gmr, sigma_gmr)
        mu_force_kmp, sigma_force_kmp = kmp.predict(x_test)
        mu_kmp_p_test.append(mu_force_kmp)
        sigma_kmp_p_test.append(sigma_force_kmp)
    mu_kmp_p_test = np.array(mu_kmp_p_test)
    sigma_kmp_p_test = np.array(sigma_kmp_p_test)

    # Equation 46 in the paper
    mu_force_kmp_test = []
    for i in range(x_kmp_test.shape[1]):
        sigma = [np.linalg.inv(sigmas[:, :, i]) for sigmas in sigma_kmp_p_test]
        sigma_sum_inv = np.linalg.inv(np.sum(sigma, axis=0))
        sigma_times_mu = [np.linalg.inv(sigmas[:, :, i])@mus[:, i] for sigmas, mus in zip(sigma_kmp_p_test, mu_kmp_p_test)]
        sigma_times_mu_sum = np.sum(sigma_times_mu, axis=0)
        mu_force_kmp_test.append(sigma_sum_inv @ sigma_times_mu_sum)
    #mu_force_kmp_test = np.array(mu_force_kmp_test).T



    x_kmp_test = x_kmp_test2

    # An are the rotation matrices for the start and end frame
    An = np.array([np.linalg.inv(np.eye(6)), np.linalg.inv(np.eye(6))])
    # bn are the translation vectors for the start and end frame
    bn = np.array([[*start_pose2, 0, 0, 0], [*end_pose2, 0, 0, 0]])
    X_force = extract_input_data(datasets)
    # X_p are the local demonstration databases
    X_p = [np.array([A @ (row - b) for row in zip(*X_force.T)]) for A, b in zip(An, bn)]
    
    # Project the reference trajectory to the local dbs
    x_gmr_p = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_gmr_train)]).T for A, b in zip(An, bn)]
    
    # Extract the local reference databases
    mu_gmr_p = []
    sigma_gmr_p = []
    for X, x_gmr in zip(X_p, x_gmr_p):
        gmm = GaussianMixtureModel(n_components=10, n_demos=H)
        gmm.fit(X)
        mu_force, sigma_force = gmm.predict(x_gmr)
        mu_gmr_p.append(mu_force)
        sigma_gmr_p.append(sigma_force)

    # Project the input into the local frames
    x_kmp_p_train = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_kmp_train)]).T for A, b in zip(An, bn)]
    x_kmp_p_test = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_kmp_test)]).T for A, b in zip(An, bn)]

    # Run KMP on both reference databases
    mu_kmp_p_test = []
    sigma_kmp_p_test = []
    kmp = KMP(l=1e-5, alpha=1e4, sigma_f=1e5, time_driven_kernel=False)
    for x_train, x_test, mu_gmr, sigma_gmr in zip(x_kmp_p_train, x_kmp_p_test, mu_gmr_p, sigma_gmr_p):
        kmp.fit(x_train, mu_gmr, sigma_gmr)
        mu_force_kmp, sigma_force_kmp = kmp.predict(x_test)
        mu_kmp_p_test.append(mu_force_kmp)
        sigma_kmp_p_test.append(sigma_force_kmp)
    mu_kmp_p_test = np.array(mu_kmp_p_test)
    sigma_kmp_p_test = np.array(sigma_kmp_p_test)

    # Equation 46 in the paper
    mu_force_kmp_test = []
    for i in range(x_kmp_test.shape[1]):
        sigma = [np.linalg.inv(sigmas[:, :, i]) for sigmas in sigma_kmp_p_test]
        sigma_sum_inv = np.linalg.inv(np.sum(sigma, axis=0))
        sigma_times_mu = [np.linalg.inv(sigmas[:, :, i])@mus[:, i] for sigmas, mus in zip(sigma_kmp_p_test, mu_kmp_p_test)]
        sigma_times_mu_sum = np.sum(sigma_times_mu, axis=0)
        mu_force_kmp_test.append(sigma_sum_inv @ sigma_times_mu_sum)
    mu_force_kmp_test = np.array(mu_force_kmp_test).T

    if False:
        # Set some waypoints
        wp_width = 10 # How many points before and after the middle to consider
        var = 1e-8 # Variance to achieve
        Fz = 0 # Force in z to achieve

        middle_index = x_kmp.shape[1] // 2
        waypoints = x_kmp[:, middle_index - wp_width : middle_index + wp_width + 1]
        N_points = waypoints.shape[1]
        forces = np.tile(np.array([0.0, 0.0, Fz]).reshape(-1,1), (1, N_points))
        sigmas = np.repeat(var*np.eye(3)[:, :, np.newaxis], N_points, axis=2)

        kmp.set_waypoint(waypoints, forces, sigmas)

    fig_vs, ax = plt.subplots(2, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)

    x_kmp_test = np.hstack((x_kmp_test1, x_kmp_test2))

    t_gmr = dt * np.arange(0,x_gmr_train.shape[1])
    t_kmp_test = dt * (x_kmp_test.shape[1] / N) * np.arange(0,x_kmp_test.shape[1])
    t_kmp_mu = dt * (mu_force_kmp_test.shape[1] / N) * np.arange(0,mu_force_kmp_test.shape[1])
    for i in range(3):
        ax[0,i].plot(t_gmr, x_gmr_train[i, :],color="red")
        ax[0,i].plot(t_kmp_test, x_kmp_test[i,:], color="green")
        ax[1,i].plot(t_gmr, mu_force[i, :],color="red")
        ax[1,i].plot(t_kmp_mu, mu_force_kmp_test[i, :],color="green")

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