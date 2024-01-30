import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib

from matplotlib.patches import Ellipse
from os.path import dirname, abspath, join
from colect.dataset import load_datasets, as_array
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
subsample = 25


def main():

    # Load the demonstrations
    datasets = load_datasets("demonstrations/experiment3/training")
    

    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]

    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    zs = np.array([[p.z for p in dataset] for dataset in datasets])
    z_mean = [np.mean(z) for z in zs]
    print(f"Delta z mean top: {np.mean(z_mean[:5])}, Delta z mean middle: {np.mean(z_mean[5:])}")
    fzs = np.array([[p.fz for p in dataset] for dataset in datasets])
    fz_mean = [np.mean(fz) for fz in fzs]
    print(f"fz mean top: {np.mean(fz_mean[:5])}, fz mean middle: {np.mean(fz_mean[5:])}")
    fz_std = [np.std(fz) for fz in fzs]


    dt = 0.001
    demo_duration = dt * (N + 1)  # Duration of each demonstration

    # Use the average pose as input for GMR prediction
    x_gmr = np.mean(zs, axis=0).reshape(-1,1).T
    
    # Prepare data for GMM/GMR
    X_force = extract_input_data(datasets)

    # GMM/GMR on the force
    gmm = GaussianMixtureModel(n_components=10, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x_gmr)

    fig_vs, ax = plt.subplots(1, 2, figsize=(16, 8))
    l_ = 1e-2
    alpha_ = 1e2
    sigma_f_ = 500
    z_ = 0.005
        
    ax[0].clear()
    ax[1].clear()
    start_pose = np.array([-0.365, 0.290, -z_])
    end_pose = np.array([-0.465, 0.290, -z_])
    x_kmp2 = linear_traj(start_pose, end_pose, n_points=200)[:,2].T
    # KMP on the force
    kmp = KMP(l=l_, alpha=alpha_, sigma_f=sigma_f_, verbose=True, time_driven_kernel=False)
    kmp.fit(x_gmr, mu_force, sigma_force)
    joblib.dump(kmp, join(ROOT, "trained_models", "experiment3_kmp.mdl"))

    mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp2.reshape(-1,1).T)

    print(f"KMP params: l: {l_}, a: {alpha_}, s: {sigma_f_}")
    print(f"KMP KL divergence: {kmp.kl_divergence}")
    print(f"Predicted force z mean: {np.mean(mu_force_kmp[0,:])}, std: {np.mean(sigma_force_kmp[0, 0,:])}")
    test = np.abs(np.array(z_mean) - (-z_))
    expected = np.argmin(test)
    print(f"Expected force z mean: {fz_mean[expected]}, std: {fz_std[expected]}")

    t_gmr = dt * np.arange(1,N+1)
    
    # Plot KMP vs GMR
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
        
    t_kmp = dt * (x_gmr.shape[1]/x_kmp2.shape[0]) * np.arange(0, x_kmp2.shape[0])
    ax[0].plot(t_kmp, x_kmp2, color="green")
    #ax[1].plot(t_gmr, mu_force[0, :],color="red")
    #ax[1,i].fill_between(x=t_gmr, y1=mu_force[i, :]+np.sqrt(sigma_force[i, i, :]), y2=mu_force[i, :]-np.sqrt(sigma_force[i, i, :]),color="red",alpha=0.35)        
    ax[1].plot(t_kmp, mu_force_kmp[0, :],color="green")
    ax[1].fill_between(x=t_kmp, y1=mu_force_kmp[0, :]+np.sqrt(sigma_force_kmp[0, 0, :]), y2=mu_force_kmp[0, :]-np.sqrt(sigma_force_kmp[0, 0, :]),color="green",alpha=0.35)        
        
    fig_vs.suptitle("Experiment 3")
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
    
    pos = np.vstack([p.z for dataset in datasets for p in dataset]).T
    force = np.vstack([p.fz for dataset in datasets for p in dataset]).T

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
    z = [p.z for p in demonstration]
    fz = [p.fz for p in demonstration]
    data = [z, fz]
    y_labels = [
        "z [m]",
        "$F_z$ [N]",
    ]
    # Plot everything
    for i in range(2):
        ax[i].plot(time, data[i], linewidth=0.6, color="grey")
        ax[i].set_ylabel(y_labels[i])
        ax[i].grid(True)
        ax[i].set_xlabel("Time [s]")

if __name__ == "__main__":
    main()