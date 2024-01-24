import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import joblib
import time
import wandb

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

ROOT = (dirname(dirname(abspath(__file__))))

# One every 'subsample' data points will be kept for each demonstration
subsample = 10

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
    X = np.vstack((pos, force))

    # The shape of the dataset should be (n_samples, n_features)
    return X.T

# Define objective/training function for KMP
def kmp_objective(config):
    # Load the demonstrations
    datasets = load_datasets("demonstrations/experiment4")
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
    x_gmr = np.mean(poses, axis=0)[:, [2,3,4]].T
    
    # Prepare data for GMM/GMR
    X_force = extract_input_data(datasets)

    # Generate the trajectory
    start_pose = np.array([-0.365, 0.290, 0.05])
    end_pose = np.array([-0.415, 0.290, 0.05])
    x_kmp_test = linear_traj(start_pose, end_pose, n_points=N//2)[:, :3].T

    np.save(join(ROOT, "trained_models", "experiment4_traj.npy"), x_kmp_test)

    # An are the rotation matrices for the start and end frame
    An = np.array([np.eye(6), np.eye(6)])
    # bn are the translation vectors for the start and end frame
    bn = np.array([[-0.365, 0.290, 0.05, 0, 0, 0], [-0.465, 0.290, 0.05, 0, 0, 0]])
    X_force = extract_input_data(datasets)
    # X_p are the local demonstration databases
    X_p = [np.array([A @ (row - b) for row in zip(*X_force.T)]) for A, b in zip(An, bn)]
    
    # Project the reference trajectory to the local dbs
    x_gmr_p = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_gmr)]).T for A, b in zip(An, bn)]
    
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
    x_kmp_p = [np.array([A[:3, :3] @ (row - b[:3]) for row in zip(*x_kmp_test)]).T for A, b in zip(An, bn)]

    # Run KMP on both reference databases
    mu_kmp_p = []
    sigma_kmp_p = []
    kl_divs = []
    for x_gmr, x_kmp in zip(x_gmr_p, x_kmp_p):
        kmp = KMP(l=config.l, alpha=config.alpha, sigma_f=config.sigma_f, time_driven_kernel=False)
        kmp.fit(x_gmr, mu_force, sigma_force)
        mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp, compute_KL=True)
        mu_kmp_p.append(mu_force_kmp)
        sigma_kmp_p.append(sigma_force_kmp)
        kl_divs.append(kmp.kl_divergence)
    mu_kmp_p = np.array(mu_kmp_p)
    sigma_kmp_p = np.array(sigma_kmp_p)

    # Equation 46 in the paper
    mu_force_kmp = []
    for i in range(x_kmp_test.shape[1]):
        sigma = [np.linalg.inv(sigmas[:, :, i]) for sigmas in sigma_kmp_p]
        sigma_sum_inv = np.linalg.inv(np.sum(sigma, axis=0))
        sigma_times_mu = [np.linalg.inv(sigmas[:, :, i])@mus[:, i] for sigmas, mus in zip(sigma_kmp_p, mu_kmp_p)]
        sigma_times_mu_sum = np.sum(sigma_times_mu, axis=0)
        mu_force_kmp.append(sigma_sum_inv @ sigma_times_mu_sum)
    mu_force_kmp = np.array(mu_force_kmp).T

    return kmp.l, kmp.alpha, kmp.sigma_f, np.mean(np.array(kl_divs))

def kmp_main():
    def agent():
        wandb.init(project="CoLeCT-KMP")
        l, alpha, sigma_f, kl_div = kmp_objective(wandb.config)
        wandb.log({"l": l, "alpha": alpha, "sigma_f": sigma_f, "kl_div": kl_div})
    return agent

# Define the search spaces for KMP
kmp_sweep_configuration = {
    "name": "force",
    "method": "grid",
    "parameters": {
        "l": {"values": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4]},
        "alpha": {"values": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4]},
        "sigma_f": {"values": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4]},
    },
}

# Start the sweep for GMM

sweep_id = wandb.sweep(sweep=kmp_sweep_configuration, project="CoLeCT-KMP")

wandb.agent(sweep_id, function=kmp_main())