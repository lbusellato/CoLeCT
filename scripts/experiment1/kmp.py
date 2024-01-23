import copy
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

from os.path import dirname, abspath, join
from colect.dataset import load_datasets, as_array
from colect.mixture import GaussianMixtureModel
from colect.kmp import KMP

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
    datasets = load_datasets("demonstrations/experiment1")
    datasets = datasets[:5]
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]

    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    dt = 0.001
    demo_duration = dt * (N + 1)  # Duration of each demonstration

    # Use the average pose as input for GMR prediction
    dataset_arrs = [as_array(dataset) for dataset in datasets]
    poses = np.stack((dataset_arrs))
    x = np.mean(poses, axis=0)[:, [2,3,4,9,10,11]].T
    np.save(join(ROOT, "trained_models", "experiment1_qa"), datasets[0][0].rot)
    np.save(join(ROOT, "trained_models", "experiment1_traj"), x)
    
    # Prepare data for GMM/GMR
    X_force = extract_input_data(datasets)

    # GMM/GMR on the force
    gmm = GaussianMixtureModel(n_components=10, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x)

    # KMP on the force
    kmp = KMP(l=5e-3, alpha=2e3, sigma_f=750, time_driven_kernel=False)
    kmp.fit(x, mu_force, sigma_force)

    joblib.dump(kmp, join(ROOT, "trained_models", "experiment1_kmp.mdl"))
    
    mu_force_kmp, sigma_force_kmp = kmp.predict(x, compute_KL=True)
    print(f"KL Divergence: {kmp.kl_divergence}")
    
    elapsed = []
    for i in range(x.shape[1]):
        start_time = time.perf_counter()
        _, _ = kmp.predict(x[:, i])
        elapsed.append(time.perf_counter() - start_time)
    elapsed = np.array(elapsed)
    print(f"Avg prediction time: {round(np.mean(elapsed),4)}±{round(np.std(elapsed),4)} s")

    t_gmr = 20.0/N * np.arange(1,N+1)
    
    # Plot KMP vs GMR
    fig_vs, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
        
    for i in range(3):
        ax[0,i].plot(t_gmr, x[i,:], color="green")
        ax[1,i].plot(t_gmr, x[i + 3,:], color="green")
        ax[2,i].plot(t_gmr, mu_force[i, :],color="red")
        ax[2,i].plot(t_gmr, mu_force_kmp[i, :],color="green")
        
    fig_vs.suptitle("Experiment 2")
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
    rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset]).T
    force = np.vstack([p.force for dataset in datasets for p in dataset]).T

    #if field != "force":
    # Compute the derivatives of the outputs
    dY = np.split(copy.deepcopy(force), H, axis=1)
    for y in dY:
        y[0, :] = np.gradient(y[0, :]) / dt
        y[1, :] = np.gradient(y[1, :]) / dt
        y[2, :] = np.gradient(y[2, :]) / dt
    dY = np.hstack(dY)

    # Add the derivatives to the dataset
    X = np.vstack((pos, rot, force))#, dY))

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
    #time = 0.001 * np.arange(1, len(time) + 1)
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