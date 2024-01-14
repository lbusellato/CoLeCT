import logging
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from os.path import dirname, abspath
from colect.dataset import load_datasets, as_array
from colect.mixture import GaussianMixtureModel
from colect.kmp import KMP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)

ROOT = dirname(dirname(abspath(__file__)))

# One every 'subsample' data points will be kept for each demonstration
subsample = 3


def main():
    """Showcase on how to set up and use GMM/KMP."""

    # Load the demonstrations
    datasets = load_datasets("demonstrations/top_vase")
    datasets = datasets[:5]
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]

    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    X_force = extract_input_data(datasets, "force")

    dt = 0.001
    demo_duration = dt * (N + 1)  # Duration of each demonstration
    x_gmr = dt * np.arange(1, N + 2).reshape(1, -1)

    # Use the average pose as input for GMR prediction
    dataset_arrs = [as_array(dataset[::subsample]) for dataset in datasets]
    poses = np.stack((dataset_arrs))
    x_kmp = np.mean(poses, axis=0)[:, [2,3,4,9,10,11]].T

    # GMM/GMR on the force
    gmm = GaussianMixtureModel(n_components=5, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x_kmp)

    # KMP on the force
    kmp = KMP(l=5e-3, alpha=2e3, sigma_f=750, verbose=True)
    kmp.fit(x_kmp, mu_force, sigma_force)
    mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp)
    

    time = [p.time for p in datasets[0]] 
    time = dt * np.arange(1, len(time) + 1)

    t_gmr = dt * (334 / 112) *np.arange(1,mu_force.shape[1] + 1)

    # Plot GMR
    fig_gmr, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
        
    for i in range(3):
        ax[2,i].plot(t_gmr, mu_force[i, :],color="red")
        
    fig_gmr.suptitle("Single point task - GMR")
    fig_gmr.tight_layout()
    
    # Plot KMP
    fig_kmp, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
    t_kmp = t_gmr
    for i in range(3):
        ax[2,i].plot(t_kmp, mu_force_kmp[i, :],color="green")
            
    fig_kmp.suptitle("Single point task - KMP")
    fig_kmp.tight_layout()
    
    # Plot KMP vs GMR
    fig_vs, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
        
    for i in range(3):
        ax[2,i].plot(t_gmr, mu_force[i, :],color="red")
        ax[2,i].plot(t_kmp, mu_force_kmp[i, :],color="green")
        
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig_vs.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.5, 0.95))

    fig_vs.suptitle("Single point task - GMR vs KMP")
    fig_vs.tight_layout()

    plt.show()


def extract_input_data(datasets: np.ndarray, field: str, dt: float = 0.1) -> np.ndarray:
    """Creates the input array for training the GMMs

    Parameters
    ----------
    datasets : np.ndarray
        The demonstration dataset.
    field : str
        Either `"position"`, `"orientation"` or `"force"`.
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

    # Input vector for time-based GMM
    x_gmr = dt * np.arange(1, N + 1).reshape(1, -1)
    X = np.tile(x_gmr, H).reshape(1, -1)

    # Depending on the requested field, set up the output vectors
    if field == "position":
        Y = np.vstack([p.position for dataset in datasets for p in dataset]).T
    elif field == "orientation":
        Y = np.vstack([p.rot_eucl for dataset in datasets for p in dataset]).T
    elif field == "force":
        # Pose-based case
        pos = np.vstack([p.position for dataset in datasets for p in dataset]).T
        rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset]).T
        X = np.vstack((pos, rot))
        Y = np.vstack([p.force for dataset in datasets for p in dataset]).T

    # Add the outputs to the dataset
    X = np.vstack((X, Y))

    #if field != "force":
    # Compute the derivatives of the outputs
    Y = np.split(Y, H, axis=1)
    for y in Y:
        y[0, :] = np.gradient(y[0, :]) / dt
        y[1, :] = np.gradient(y[1, :]) / dt
        y[2, :] = np.gradient(y[2, :]) / dt
    dY = np.hstack(Y)

    # Add the derivatives to the dataset
    X = np.vstack((X, dY))

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
    qx = [p.rot_eucl[0] for p in demonstration]
    qy = [p.rot_eucl[1] for p in demonstration]
    qz = [p.rot_eucl[2] for p in demonstration]
    fx = [p.fx for p in demonstration]
    fy = [p.fy for p in demonstration]
    fz = [p.fz for p in demonstration]
    data = [x, y, z, qx, qy, qz, fx, fy, fz]
    y_labels = [
        "x [m]",
        "y [m]",
        "z [m]",
        "$q_{ex}$",
        "$q_{ey}$",
        "$q_{ez}$",
        "$F_x$ [N]",
        "$F_y$ [N]",
        "$F_z$ [N]",
    ]
    # Plot everything
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(time, data[i * 3 + j], linewidth=0.6, color="grey")
            ax[i, j].set_ylabel(y_labels[i * 3 + j])
            ax[i, j].grid(True)
            if i == 2:
                ax[i, j].set_xlabel("Time [s]")

# ugly ugly ugly ugly ugly
def plot_demo_no_force(
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

    time = [p.time for p in demonstration] #np.arange(dt, duration - dt, dt)
    # Recover data
    x = [p.x for p in demonstration]
    y = [p.y for p in demonstration]
    z = [p.z for p in demonstration]
    qx = [p.rot_eucl[0] for p in demonstration]
    qy = [p.rot_eucl[1] for p in demonstration]
    qz = [p.rot_eucl[2] for p in demonstration]
    data = [x, y, z, qx, qy, qz]
    y_labels = [
        "x [m]",
        "y [m]",
        "z [m]",
        "$q_{ex}$",
        "$q_{ey}$",
        "$q_{ez}$",
    ]
    # Plot everything
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(time, data[i * 3 + j], linewidth=0.6, color="grey")
            ax[i, j].set_ylabel(y_labels[i * 3 + j])
            ax[i, j].grid(True)
            if i == 1:
                ax[i, j].set_xlabel("Time [s]")


if __name__ == "__main__":
    main()
