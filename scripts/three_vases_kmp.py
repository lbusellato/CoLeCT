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

ROOT = dirname(dirname(abspath(__file__)))

# One every 'subsample' data points will be kept for each demonstration
subsample = 10


def main():
    """Showcase on how to set up and use GMM/KMP."""

    # Load the demonstrations
    datasets = load_datasets("demonstrations/three_vases")
    
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]

    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    zs = np.array([[p.z for p in dataset] for dataset in datasets])
    z_mean = [np.mean(z) for z in zs]
    fzs = np.array([[p.fz for p in dataset] for dataset in datasets])
    fz_mean = [np.mean(fz) for fz in fzs]
    fz_std = [np.std(fz) for fz in fzs]


    dt = 0.001
    demo_duration = dt * (N + 1)  # Duration of each demonstration

    # Use the average pose as input for GMR prediction
    x_gmr = np.array([[p.z for p in dataset] for dataset in datasets]).flatten().reshape(-1,1).T
    #x_gmr = np.mean(zs, axis=0).reshape(-1,1).T
    
    # Prepare data for GMM/GMR
    X_force = extract_input_data(datasets)

    qa = datasets[0][0].rot
    quat = datasets[0][0].rot
    start_pose = np.array([-1.45, 0., -0.01,quat[0],quat[1],quat[2],quat[3]])
    end_pose = np.array([-1.35, 0., -0.01,quat[0],quat[1],quat[2],quat[3]])
    x_kmp = linear_traj(start_pose, end_pose, n_points=N, qa=qa).T
    x_kmp = x_kmp[2,:]
    np.save(join(ROOT, "trained_models", "traj.py"), x_kmp)

    # GMM/GMR on the force
    gmm = GaussianMixtureModel(n_components=5, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x_gmr)

    plt.ion()
    fig_vs, ax = plt.subplots(1, 2, figsize=(16, 8))
    l_ = 0.05
    alpha_ = 5e3
    sigma_f_ = 1e3
    z_ = 0.01
    try:
        while True:
            l = input("l: ")
            l_ = float(l) if l != '' else l_
            alpha = input("a: ")
            alpha_ = float(alpha) if alpha != '' else alpha_
            sigma_f = input("s: ")
            sigma_f_ = float(sigma_f) if sigma_f != '' else sigma_f_
            z = input("z: ")
            z_ = float(z) if z != '' else z_
            ax[0].clear()
            ax[1].clear()
            start_pose = np.array([-1.45, 0., -z_,quat[0],quat[1],quat[2],quat[3]])
            end_pose = np.array([-1.35, 0., -z_,quat[0],quat[1],quat[2],quat[3]])
            x_kmp = linear_traj(start_pose, end_pose, n_points=N, qa=qa).T
            x_kmp = x_kmp[2,:]
            # KMP on the force
            kmp = KMP(l=l_, alpha=alpha_, sigma_f=sigma_f_, verbose=True, time_driven_kernel=False)
            kmp.fit(x_gmr, mu_force, sigma_force)

            joblib.dump(kmp, join(ROOT, "trained_models", "kmp.mdl"))

            mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp.reshape(-1,1).T)

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
                
            ax[0].plot(t_gmr, x_kmp, color="green")
            #ax[1].plot(t_gmr, mu_force[0, :],color="red")
            #ax[1,i].fill_between(x=t_gmr, y1=mu_force[i, :]+np.sqrt(sigma_force[i, i, :]), y2=mu_force[i, :]-np.sqrt(sigma_force[i, i, :]),color="red",alpha=0.35)        
            ax[1].plot(t_gmr, mu_force_kmp[0, :],color="green")
            ax[1].fill_between(x=t_gmr, y1=mu_force_kmp[0, :]+np.sqrt(sigma_force_kmp[0, 0, :]), y2=mu_force_kmp[0, :]-np.sqrt(sigma_force_kmp[0, 0, :]),color="green",alpha=0.35)        
                
            fig_vs.suptitle("Three vase sliding task - GMR vs KMP")
            fig_vs.tight_layout()

            plt.show(block=False)
    except KeyboardInterrupt:
        pass

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
    #rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset]).T
    force = np.vstack([p.fz for dataset in datasets for p in dataset]).T

    #if field != "force":
    # Compute the derivatives of the outputs
    dY = np.split(copy.deepcopy(force), H, axis=1)
    for y in dY:
        y[0, :] = np.gradient(y[0, :]) / dt
    dY = np.hstack(dY)

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
