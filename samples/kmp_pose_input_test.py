import logging
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from os.path import dirname, abspath
from src.dataset import load_datasets
from src.mixture import GaussianMixtureModel
from src.kmp import KMP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)

ROOT = dirname(dirname(abspath(__file__)))

# One every 'subsample' data points will be kept for each demonstration
subsample = 100


def main():
    """Showcase on how to set up and use GMM/KMP."""

    # Load the demonstrations
    datasets = load_datasets("demonstrations/single_point_task")
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]

    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    # Create the training datasets used to train the GMMs
    X_pos = extract_input_data(datasets, "position")
    X_rot = extract_input_data(datasets, "orientation")
    X_force = extract_input_data(datasets, "force")

    # Create the input vector for time-based GMR
    gmm_dt = 0.1
    x_gmr = gmm_dt * np.arange(1, N + 2).reshape(1, -1)

    demo_duration = gmm_dt * (N + 1)  # Duration of each demonstration

    # Create the input vector for time-based KMP
    kmp_dt = 0.1
    x_kmp = np.arange(kmp_dt, demo_duration, kmp_dt).reshape(1, -1)

    # GMM/GMR on the position
    gmm = GaussianMixtureModel(n_components=5, n_demos=H)
    gmm.fit(X_pos)
    mu_pos_gmm = gmm.means
    sigma_pos_gmm = gmm.covariances
    mu_pos, sigma_pos = gmm.predict(x_gmr)

    # GMM/GMR on the orientation
    gmm = GaussianMixtureModel(n_components=3, n_demos=H)
    gmm.fit(X_rot)
    mu_rot_gmm = gmm.means
    sigma_rot_gmm = gmm.covariances
    mu_rot, sigma_rot = gmm.predict(x_gmr)

    # Create the input vector for pose-based GMR -> get the poses predicted by the previous GMRs
    x_gmr_pose = np.vstack((mu_pos[:3, :], mu_rot[:3, :]))

    # GMM/GMR on the force
    gmm = GaussianMixtureModel(n_components=5, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x_gmr_pose)

    # KMP on the position
    kmp = KMP(l=0.5, sigma_f=5, verbose=True)
    kmp.fit(x_gmr, mu_pos, sigma_pos)
    mu_pos_kmp, sigma_pos_kmp = kmp.predict(x_kmp)

    # KMP on the orientation
    kmp = KMP(l=0.5, sigma_f=5, verbose=True)
    kmp.fit(x_gmr, mu_rot, sigma_rot)
    mu_rot_kmp, sigma_rot_kmp = kmp.predict(x_kmp)

    # Create the input vector for pose-based KMP -> get the poses predicted by the previous KMPs
    x_kmp_pose = np.vstack((mu_pos_kmp[:3, :], mu_rot_kmp[:3, :]))

    # KMP on the force
    kmp = KMP(l=5e-3, alpha=2e3, sigma_f=750, verbose=True)
    kmp.fit(x_gmr_pose, mu_force, sigma_force)
    mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp_pose)
    
    # Plot GMM
    fig_gmm, ax = plt.subplots(2, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo_no_force(ax, dataset, demo_duration)
    t_gmr = x_gmr.flatten()
    # ugly ugly ugly
    for i in range(3):
        for c in range(sigma_pos_gmm.shape[2]):
            cov_pos = np.array([[sigma_pos_gmm[0,0,c],sigma_pos_gmm[0,i+1,c]],
                                [sigma_pos_gmm[i+1,0,c],sigma_pos_gmm[i+1,i+1,c]]])
            e_vals, e_vecs = np.linalg.eig(cov_pos)
            major_axis = 2 * np.sqrt(e_vals[0]) * e_vecs[:, 0]
            minor_axis = 2 * np.sqrt(e_vals[1]) * e_vecs[:, 1]
            angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
            pos = [mu_pos_gmm[0,c],mu_pos_gmm[i+1,c]] 
            width = np.linalg.norm(major_axis)
            height = np.linalg.norm(minor_axis)
            ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, facecolor='orange', alpha=0.6,zorder=1)
            ax[0,i].add_artist(ellipse)
        ax[0,i].scatter(mu_pos_gmm[0],mu_pos_gmm[i+1],zorder=3,color="blue")
        for c in range(sigma_rot_gmm.shape[2]):
            cov_rot = np.array([[sigma_rot_gmm[0,0,c],sigma_rot_gmm[0,i+1,c]],
                                [sigma_rot_gmm[i+1,0,c],sigma_rot_gmm[i+1,i+1,c]]])
            e_vals, e_vecs = np.linalg.eig(cov_rot)
            major_axis = 2 * np.sqrt(e_vals[0]) * e_vecs[:, 0]
            minor_axis = 2 * np.sqrt(e_vals[1]) * e_vecs[:, 1]
            angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
            pos = [mu_rot_gmm[0,c],mu_rot_gmm[i+1,c]] 
            width = np.linalg.norm(major_axis)
            height = np.linalg.norm(minor_axis)
            ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, facecolor='orange', alpha=0.6,zorder=1)
            ax[1,i].add_artist(ellipse)
        ax[1,i].scatter(mu_rot_gmm[0],mu_rot_gmm[i+1],zorder=3,color="blue")
        ax[0,i].set_xlim(left=0,right=t_gmr[-1])
        ax[1,i].set_xlim(left=0,right=t_gmr[-1])
        ax[0,i].set_autoscale_on(True)
        ax[0,i].autoscale_view(True,True,True)
        ax[1,i].set_autoscale_on(True)
        ax[1,i].autoscale_view(True,True,True)
    
    ax[0,2].set_ylim(top=1.1,bottom=1.09)
    ax[1,1].set_ylim(top=0.0015,bottom=-0.0015)
    ax[1,2].set_ylim(top=0.0035,bottom=-0.0035)
        
    fig_gmm.suptitle("Single point task - GMM")
    fig_gmm.tight_layout()

    # Plot GMR
    fig_gmr, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
    t_gmr = x_gmr.flatten()
    for i in range(3):
        ax[0,i].fill_between(x=t_gmr, y1=mu_pos[i, :]+np.sqrt(sigma_pos[i, i, :]), y2=mu_pos[i, :]-np.sqrt(sigma_pos[i, i, :]),color="red",alpha=0.35)        
        ax[1,i].fill_between(x=t_gmr, y1=mu_rot[i, :]+np.sqrt(sigma_rot[i, i, :]), y2=mu_rot[i, :]-np.sqrt(sigma_rot[i, i, :]),color="red",alpha=0.35)
        ax[0,i].plot(t_gmr, mu_pos[i, :],color="red")
        ax[1,i].plot(t_gmr, mu_rot[i, :],color="red")
        ax[2,i].plot(t_gmr, mu_force[i, :],color="red")
        
    fig_gmr.suptitle("Single point task - GMR")
    fig_gmr.tight_layout()
    
    # Plot KMP
    fig_kmp, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
    t_kmp = x_kmp.flatten()
    for i in range(3):
        ax[0,i].fill_between(x=t_kmp, y1=mu_pos_kmp[i, :]+np.sqrt(sigma_pos_kmp[i, i, :]), y2=mu_pos_kmp[i, :]-np.sqrt(sigma_pos_kmp[i, i, :]),color="green",alpha=0.35)        
        ax[1,i].fill_between(x=t_kmp, y1=mu_rot_kmp[i, :]+np.sqrt(sigma_rot_kmp[i, i, :]), y2=mu_rot_kmp[i, :]-np.sqrt(sigma_rot_kmp[i, i, :]),color="green",alpha=0.35)
        ax[0,i].plot(t_kmp, mu_pos_kmp[i, :],color="green")
        ax[1,i].plot(t_kmp, mu_rot_kmp[i, :],color="green")
        ax[2,i].plot(t_kmp, mu_force_kmp[i, :],color="green")
            
    fig_kmp.suptitle("Single point task - KMP")
    fig_kmp.tight_layout()
    
    # Plot KMP vs GMR
    fig_vs, ax = plt.subplots(3, 3, figsize=(16, 8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
    t_kmp = x_kmp.flatten()
    for i in range(3):
        ax[0,i].plot(t_gmr, mu_pos[i, :],color="red", label="GMR")
        ax[1,i].plot(t_gmr, mu_rot[i, :],color="red")
        ax[2,i].plot(t_gmr, mu_force[i, :],color="red")
        ax[0,i].plot(t_kmp, mu_pos_kmp[i, :],color="green", label="KMP")
        ax[1,i].plot(t_kmp, mu_rot_kmp[i, :],color="green")
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

    time = np.arange(dt, duration - dt, dt)
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

    time = np.arange(dt, duration - dt, dt)
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
