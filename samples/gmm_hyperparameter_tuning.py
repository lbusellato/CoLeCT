import copy
import numpy as np
import wandb

from os.path import dirname, abspath
from src.dataset import load_datasets
from src.mixture import GaussianMixtureModel

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
Y_force = np.vstack([p.force for dataset in datasets for p in dataset[::subsample]]).T
# Compute the derivatives of the forces
forces = np.split(copy.deepcopy(Y_force), H, axis=1)
for force in forces:
    force[0,:] = np.gradient(force[0,:])/gmm_dt
    force[1,:] = np.gradient(force[1,:])/gmm_dt
    force[2,:] = np.gradient(force[2,:])/gmm_dt
dY_force = np.hstack(forces)
X_force = np.vstack((Y_pos, Y_rot, Y_force, dY_force))

# 1: Define objective/training function
def objective(data, config):
    gmm = GaussianMixtureModel(
        n_components=config.n_components, n_demos=5)
    gmm.fit(data)
    return gmm.n_components, gmm.bic(data)

def main(data):
    def agent():
        wandb.init(project="CoLeCT-GMM")
        n_comp, bic = objective(data, wandb.config)
        wandb.log({"n_components": n_comp, "bic": bic})
    return agent

# 2: Define the search spaces
sweep_configuration = {
    "name": "",
    "method": "grid",
    "parameters": {
        "n_components": {"values": [i for i in range(1,21)]}
    },
}

# 3: Start the sweeps
sweep_configuration['name'] = 'position'
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CoLeCT-GMM")

wandb.agent(sweep_id, function=main(X_pos.T))

sweep_configuration['name'] = 'orientation'
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CoLeCT-GMM")

wandb.agent(sweep_id, function=main(X_rot.T))

sweep_configuration['name'] = 'force'
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CoLeCT-GMM")

wandb.agent(sweep_id, function=main(X_force.T))