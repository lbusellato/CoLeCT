import copy
import numpy as np
import wandb

from os.path import dirname, abspath
from colect.dataset import load_datasets, as_array
from colect.datatypes import Quaternion
from colect.mixture import GaussianMixtureModel
from colect.kmp import KMP
from colect.utils import linear_traj

wandb.login()

ROOT = dirname(dirname(abspath(__file__)))
subsample = 10
    
# Load the demonstrations
datasets = load_datasets('demonstrations/top_vase')
datasets = datasets[:5]
# Prepare the data for GMM/GMR
H = len(datasets)  # Number of demonstrations
N = len(datasets[0]) // subsample  # Length of each demonstration
gmm_dt = 0.1
# Prepare data for GMM/GMR
Y_pos = np.vstack(
    [p.position for dataset in datasets for p in dataset[::subsample]]).T
Y_rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset[::subsample]]).T
Y_force = np.vstack([p.force for dataset in datasets for p in dataset[::subsample]]).T
# Compute the derivatives of the forces
forces = np.split(copy.deepcopy(Y_force), H, axis=1)
for force in forces:
    force[0,:] = np.gradient(force[0,:])/gmm_dt
    force[1,:] = np.gradient(force[1,:])/gmm_dt
    force[2,:] = np.gradient(force[2,:])/gmm_dt
dY_force = np.hstack(forces)
X_force = np.vstack((Y_pos, Y_rot, Y_force, dY_force)).T

# Use the average pose as input for GMR prediction
dataset_arrs = [as_array(dataset[::subsample]) for dataset in datasets]
poses = np.stack((dataset_arrs))
x_gmr_pose = np.mean(poses, axis=0)[:, [2,3,4,9,10,11]].T

# GMM/GMR on the force
gmm = GaussianMixtureModel(n_components=10, n_demos=H)
gmm.fit(X_force)
mu_force, sigma_force = gmm.predict(x_gmr_pose)

# Data for KMP prediction -> validation
qa = datasets[0][0].rot
rot_vector = np.array([3.073, -0.652, 0.0])
quat = Quaternion.from_rotation_vector(rot_vector)
start_pose = np.array([-0.365, 0.290, 0.05,quat[0],quat[1],quat[2],quat[3]])
end_pose = np.array([-0.465, 0.290, 0.05,quat[0],quat[1],quat[2],quat[3]])
x_kmp = linear_traj(start_pose, end_pose, n_points=N, qa=qa).T

# Define objective/training function for KMP
def kmp_objective(mu, sigma, config):
    kmp = KMP(l=config.l, alpha=config.alpha, sigma_f=config.sigma_f)
    kmp.fit(x_gmr_pose, mu, sigma)
    kmp.predict(x_kmp)
    return kmp.l, kmp.alpha, kmp.sigma_f, kmp.kl_divergence

def kmp_main(mu, sigma):
    def agent():
        wandb.init(project="CoLeCT-KMP")
        l, alpha, sigma_f, kl_div = kmp_objective(mu, sigma, wandb.config)
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

wandb.agent(sweep_id, function=kmp_main(mu_force, sigma_force))