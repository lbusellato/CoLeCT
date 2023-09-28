import copy
import numpy as np
import wandb

from os.path import dirname, abspath
from src.dataset import load_datasets
from src.mixture import GaussianMixtureModel
from src.kmp import KMP

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
Y_rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset[::subsample]]).T
# Compute (approximated) relative angular velocities
quats_eucl = np.split(copy.deepcopy(Y_rot), H, axis=1)
for quat_eucl in quats_eucl:
    quat_eucl[0,:] = np.gradient(quat_eucl[0,:])/gmm_dt
    quat_eucl[1,:] = np.gradient(quat_eucl[1,:])/gmm_dt
    quat_eucl[2,:] = np.gradient(quat_eucl[2,:])/gmm_dt
dY_rot = np.hstack(quats_eucl)
X_rot = np.vstack((X, Y_rot, dY_rot))
demo_dura = gmm_dt * (N + 1) # Duration of each demonstration
kmp_dt = 0.01
x_kmp = np.arange(kmp_dt, demo_dura, kmp_dt).reshape(1, -1)

# GMM/GMR on the position
gmm = GaussianMixtureModel(n_components=5, n_demos=H, diag_reg_factor=1e-6)
gmm.fit(X_pos.T)
mu_pos, sigma_pos = gmm.predict(x_gmr)
# GMM/GMR on the orientation
gmm = GaussianMixtureModel(n_components=3, n_demos=H, diag_reg_factor=1e-6)
gmm.fit(X_rot.T)
mu_rot, sigma_rot = gmm.predict(x_gmr)

# 1: Define objective/training function
def objective(x_gmr, x_kmp, mu, sigma, config):
    kmp = KMP(l=config.l, alpha=config.alpha, sigma_f=config.sigma_f)
    kmp.fit(x_gmr, mu, sigma)
    kmp.predict(x_kmp)
    return kmp.l, kmp.alpha, kmp.sigma_f, kmp.kl_divergence

def main(x_gmr, x_kmp, mu, sigma):
    def agent():
        wandb.init(project="CoLeCT-GMM")
        l, alpha, sigma_f, kl_div = objective(x_gmr, x_kmp, mu, sigma, wandb.config)
        wandb.log({"l": l, "alpha": alpha, "sigma_f": sigma_f, "kl_div": kl_div})
    return agent

# 2: Define the search spaces
sweep_configuration = {
    "name": "",
    "method": "grid",
    "parameters": {
        "l": {"values": [0.01, 0.1, 1, 10]},
        "alpha": {"values": [0.5, 1, 10, 50]},
        "sigma_f": {"values": [0.01, 0.1, 1, 10]},
    },
}

# 3: Start the sweeps
sweep_configuration['name'] = 'position'
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CoLeCT-KMP")

wandb.agent(sweep_id, function=main(x_gmr, x_kmp, mu_pos, sigma_pos))

sweep_configuration['name'] = 'orientation'
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CoLeCT-KMP")
