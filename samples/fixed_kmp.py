import copy
import logging
import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import dirname, abspath, join, exists
from src.dataset import load_datasets
from src.datatypes import Quaternion
from src.mixture import GaussianMixtureModel
from src.kmp import KMP

from matplotlib.patches import Ellipse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

ROOT = dirname(dirname(abspath(__file__)))

def main():
    skip = True
    # Showcase GMR and KMP on the single point task dataset
    # Load the demonstrations
    datasets = load_datasets('demonstrations/single_point_task')
    subsample = 100
    if not skip:
        # Prepare the data for GMM/GMR
        H = len(datasets)  # Number of demonstrations
        N = len(datasets[0]) // subsample  # Length of each demonstration
        dt = 0.1
        x = dt*np.arange(1, N + 2).reshape(1, -1)
        demo_dura = dt*(N+1) # Time duration of each demonstration
        X = np.tile(x, H).reshape(1, -1)
        Y_pos = np.vstack([p.position for dataset in datasets for p in dataset[::subsample]]).T
        positions = np.split(copy.deepcopy(Y_pos), 10, axis=1)
        for position in positions:
            position[0,:] = np.gradient(position[0,:])/dt
            position[1,:] = np.gradient(position[1,:])/dt
            position[2,:] = np.gradient(position[2,:])/dt
        dY_pos = np.hstack(positions)
        X = np.vstack((X, Y_pos, dY_pos))
        # GMM/GMR on the position
        gmm = GaussianMixtureModel(n_components=10)
        gmm.init(X)
        gmm.fit(X)
        mu_pos, sigma_pos = gmm.predict(x)
        # KMP on the position
        kmp = KMP(l=0.5, alpha=40, sigma_f=1, verbose=True)
        kmp.fit(x, mu_pos, sigma_pos)
        dt = 0.01
        x = dt*np.arange(1, demo_dura/dt + 1).reshape(1, -1)
        mu_pos_kmp, sigma_pos_kmp = kmp.predict(x) 
        np.save(join(ROOT, "trained_models/mu_pos.npy"), mu_pos)
        np.save(join(ROOT, "trained_models/sigma_pos.npy"), sigma_pos)
        np.save(join(ROOT, "trained_models/mu_pos_kmp.npy"), mu_pos_kmp)
        np.save(join(ROOT, "trained_models/sigma_pos_kmp.npy"), sigma_pos_kmp)
    mu_pos = np.load(join(ROOT, "trained_models/mu_pos.npy"))
    sigma_pos = np.load(join(ROOT, "trained_models/sigma_pos.npy"))
    mu_pos_kmp = np.load(join(ROOT, "trained_models/mu_pos_kmp.npy"))
    sigma_pos_kmp = np.load(join(ROOT, "trained_models/sigma_pos_kmp.npy"))
    # Plot everything
    fig, ax = plt.subplots(1, 3, figsize=(16,8))
    for dataset in datasets:
        plot_demo(ax, dataset[::subsample])
    t_gmr = np.arange(0.1, 11.6, 0.1)
    t_kmp = np.arange(0.01, 11.51, 0.01)
    for i in range(3):
        # GMR
        ax[i].errorbar(x=t_gmr, y=mu_pos[i, :], yerr=np.sqrt(sigma_pos[i,i,:]), color='grey', alpha=0.3)
        ax[i].plot(t_gmr, mu_pos[i, :], color='blue')
        # KMP
        ax[i].errorbar(x=t_kmp, y=mu_pos_kmp[i, :], yerr=np.sqrt(sigma_pos_kmp[i,i,:]), color='red', alpha=0.1)
        ax[i].plot(t_kmp, mu_pos_kmp[i, :], color='red')
    fig.suptitle('Single point task - GMR and KMP')
    fig.tight_layout()
    plots_path = join(ROOT, 'media/single_point_task_kmp.png')
    plt.savefig(plots_path)
    plt.show()

def plot_demo(ax, demonstration, linestyle='solid', label=''):
    # Plot all the data in a demonstration
    time = np.arange(0.1, 11.6, 0.1)
    x = [p.x for p in demonstration]
    y = [p.y for p in demonstration]
    z = [p.z for p in demonstration]
    data = [x, y, z]
    y_labels = ['x [m]', 'y [m]', 'z [m]']
    for j in range(3):
        ax[j].plot(time, data[j], linestyle=linestyle, label=label, linewidth=0.6, color='grey')
        ax[j].set_ylabel(y_labels[j])
        ax[j].grid(True)
        ax[j].set_xlabel('Time [s]')



if __name__ == '__main__':
    main()