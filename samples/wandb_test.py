import copy
import numpy as np
import wandb

from os.path import dirname, abspath
from src.dataset import load_datasets
from src.mixture import GaussianMixtureModel
from sklearn.mixture import GaussianMixture

wandb.login()

ROOT = dirname(dirname(abspath(__file__)))
subsample = 100

# Load the demonstrations
datasets = load_datasets('demonstrations/single_point_task')
# Prepare the data for GMM/GMR
H = len(datasets)  # Number of demonstrations
N = len(datasets[0]) // subsample  # Length of each demonstration
gmm_dt = 0.1
x_gmr = gmm_dt*np.arange(1, N + 2).reshape(1, -1)
X = np.tile(x_gmr, H).reshape(1, -1)
# Prepare data for GMM/GMR
Y_pos = np.vstack(
    [p.position for dataset in datasets for p in dataset[::subsample]]).T
# Compute (approximated) linear velocities
positions = np.split(copy.deepcopy(Y_pos), H, axis=1)
for position in positions:
    position[0, :] = np.gradient(position[0, :])/gmm_dt
    position[1, :] = np.gradient(position[1, :])/gmm_dt
    position[2, :] = np.gradient(position[2, :])/gmm_dt
dY_pos = np.hstack(positions)
X_pos = np.vstack((X, Y_pos, dY_pos))

# 1: Define objective/training function
def objective(data, config):
    """gmm = GaussianMixtureModel(
        n_components=config.n_components, n_demos=config.n_demos, diag_reg_factor=config.diag_reg_factor)
    return gmm.fit(data)"""
    gmm = GaussianMixture(n_components=config.n_components, reg_covar=config.diag_reg_factor)
    gmm.fit(data)
    return gmm.bic(data)

def main():
    wandb.init(project="CoLeCT")
    bic = objective(X_pos.T, wandb.config)
    wandb.log({"bic": bic})


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "parameters": {
        "diag_reg_factor": {"values": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
        "n_components": {"values": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "n_demos": {"value": 5}
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CoLeCT")

wandb.agent(sweep_id, function=main)