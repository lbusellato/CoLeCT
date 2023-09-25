import copy
import logging
import numpy as np
import wandb

wandb.login()

from os.path import dirname, abspath
from src.dataset import load_datasets
from src.mixture import GaussianMixtureModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

# Set up wandb to track the script
sweep_config = {
    'method': 'grid'
}
parameters_dict = {
    'n_components': {
        'values': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    },
    'diag_reg_factor': {
        'values': [1, 10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7, 10e-8, 10e-9]
    },
}
sweep_config['parameters'] = parameters_dict
parameters_dict.update({'n_demos': {
    'value': 5
}})

ROOT = dirname(dirname(abspath(__file__)))
subsample = 100


def train(dataset, config=None):
    with wandb.init(config):
        config = wandb.config
        gmm = GaussianMixtureModel(
            n_components=config.n_components, n_demos=5, diag_reg_factor=config.diag_reg_factor)
        bic, avg_LL = gmm.fit(dataset)
        wandb.log({'BIC': bic, 'avg_LL': avg_LL})


def main():
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

    sweep_id = wandb.sweep(sweep_config, project='CoLeCT')
    wandb.agent(sweep_id, train(X_pos))


if __name__ == '__main__':
    main()
