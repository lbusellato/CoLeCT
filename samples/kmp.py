import logging
import matplotlib.pyplot as plt
import numpy as np

from os.path import dirname, abspath
from src.dataset import load_datasets
from src.mixture import GaussianMixtureModel
from src.kmp import KMP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

ROOT = dirname(dirname(abspath(__file__)))

# One every 'subsample' data points will be kept for each demonstration
subsample = 100


def main():
    """Showcase on how to set up and use GMM/KMP.
    """
    
    # Load the demonstrations
    datasets = load_datasets('demonstrations/single_point_task')
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[::subsample]
    
    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0]) # Length of each demonstration
    
    # Create the training datasets used to train the GMMs
    X_pos = extract_input_data(datasets, 'position')
    X_rot = extract_input_data(datasets, 'orientation')
    X_force = extract_input_data(datasets, 'force')
    
    # Create the input vector for GMR (time series)
    gmm_dt = 0.1
    x_gmr = gmm_dt*np.arange(1, N + 2).reshape(1, -1)
    
    demo_duration = gmm_dt * (N + 1) # Duration of each demonstration
    
    # Create the input vector for KMP (time series)
    kmp_dt = 0.05
    x_kmp = np.arange(kmp_dt, demo_duration, kmp_dt).reshape(1, -1)
    
    # GMM/GMR on the position
    gmm = GaussianMixtureModel(n_components=5, n_demos=H)
    gmm.fit(X_pos)
    mu_pos, sigma_pos = gmm.predict(x_gmr)
    
    # GMM/GMR on the orientation
    gmm = GaussianMixtureModel(n_components=3, n_demos=H)
    gmm.fit(X_rot)
    mu_rot, sigma_rot = gmm.predict(x_gmr)
        
    # GMM/GMR on the force
    gmm = GaussianMixtureModel(n_components=5, n_demos=H)
    gmm.fit(X_force)
    mu_force, sigma_force = gmm.predict(x_gmr)
        
    # KMP on the position
    kmp = KMP(l=0.5, alpha=40, sigma_f=1, verbose=True)
    kmp.fit(x_gmr, mu_pos, sigma_pos)
    mu_pos_kmp, sigma_pos_kmp = kmp.predict(x_kmp) 
        
    # KMP on the orientation
    kmp = KMP(l=0.5, alpha=30, sigma_f=2, verbose=True)
    kmp.fit(x_gmr, mu_rot, sigma_rot)
    mu_rot_kmp, sigma_rot_kmp = kmp.predict(x_kmp)
        
    # KMP on the force
    kmp = KMP(l=0.5, alpha=40, sigma_f=1, verbose=True)
    kmp.fit(x_gmr, mu_force, sigma_force)
    mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp)
        
    # Plot everything
    fig, ax = plt.subplots(3, 3, figsize=(16,8))
    for dataset in datasets:
        plot_demo(ax, dataset, demo_duration)
    t_gmr = x_gmr.flatten()
    t_kmp = x_kmp.flatten()
    for i in range(3):
        ax[0, i].errorbar(x=t_gmr, y=mu_pos[i, :], yerr=np.sqrt(sigma_pos[i,i,:]), color='red', alpha=0.35)
        ax[0, i].errorbar(x=t_kmp, y=mu_pos_kmp[i, :], yerr=np.sqrt(sigma_pos_kmp[i,i,:]), color='green', alpha=0.25)          
        ax[1, i].errorbar(x=t_gmr, y=mu_rot[i, :], yerr=np.sqrt(sigma_rot[i,i,:]), color='red', alpha=0.35)
        ax[1, i].errorbar(x=t_kmp, y=mu_rot_kmp[i, :], yerr=np.sqrt(sigma_rot_kmp[i,i,:]), color='green', alpha=0.25)   
        ax[2, i].errorbar(x=t_gmr, y=mu_force[i, :], yerr=np.sqrt(sigma_force[i,i,:]), color='red', alpha=0.35)
        ax[2, i].errorbar(x=t_kmp, y=mu_force_kmp[i, :], yerr=np.sqrt(sigma_force_kmp[i,i,:]), color='green', alpha=0.25)
    fig.suptitle('Single point task - GMR and KMP')
    fig.tight_layout()
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
    N = len(datasets[0]) # Length of each demonstration
    
    # Input vector for time-based GMM
    x_gmr = dt*np.arange(1, N + 1).reshape(1, -1)
    X = np.tile(x_gmr, H).reshape(1, -1)
    
    # Depending on the requested field, set up the output vectors
    if field == 'position':
        Y = np.vstack([p.position for dataset in datasets for p in dataset]).T
    elif field == 'orientation':
        Y = np.vstack([p.rot_eucl for dataset in datasets for p in dataset]).T
    elif field == 'force':
        Y = np.vstack([p.force for dataset in datasets for p in dataset]).T
        
    # Add the outputs to the dataset
    X = np.vstack((X, Y))
    
    # Compute the derivatives of the outputs
    Y = np.split(Y, H, axis=1)
    for y in Y:
        y[0,:] = np.gradient(y[0,:])/dt
        y[1,:] = np.gradient(y[1,:])/dt
        y[2,:] = np.gradient(y[2,:])/dt
    dY = np.hstack(Y)
    
    # Add the derivatives to the dataset
    X = np.vstack((X, dY))
    
    # The shape of the dataset should be (n_samples, n_features)
    return X.T

def plot_demo(ax: plt.Axes, demonstration: np.ndarray, duration: float, dt: float = 0.1):
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
    y_labels = ['x [m]', 'y [m]', 'z [m]',
                'qx', 'qy', 'qz',
                'Fx [N]', 'Fy [N]', 'Fz [N]']
    # Plot everything
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(time, data[i*3 + j], linewidth=0.6, color='grey')
            ax[i, j].set_ylabel(y_labels[i*3 + j])
            ax[i, j].grid(True)
            if i == 2:
                ax[i, j].set_xlabel('Time [s]')


if __name__ == '__main__':
    main()
