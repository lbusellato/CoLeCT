import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

from os import listdir
from os.path import dirname, abspath, join
from src.datatypes import Quaternion
from src.mixture import GaussianMixtureModel
from src.kmp import KMP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

ROOT = dirname(dirname(abspath(__file__)))
gmm_hyperparameter_tuning = False
kmp_hyperparameter_tuning = False

class single_point_task():
    def __init__(self, verbose: bool = True) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)
        # Load the demonstrations
        regex = r'dataset(\d{2})\.npy'
        datasets_path = 'demonstrations/single_point_task'
        datasets_path = join(ROOT, datasets_path)
        datasets = [f for f in listdir(
            datasets_path) if re.match(regex, f) is not None]
        datasets.sort()
        datasets = [np.load(join(datasets_path, f), allow_pickle=True)
                    for f in datasets]
        # Prepare the data for GMM/GMR
        subsample = 25
        H = len(datasets)  # Number of demonstrations
        N = len(datasets[0]) // subsample  # Length of each demonstration
        dt = 0.001
        x = dt*np.arange(1, N + 1).reshape(1, -1)
        X = dt*np.tile(np.arange(N + 1), H).reshape(1, -1)
        Y_pos = np.vstack([p.position for dataset in datasets for p in dataset[::subsample]]).T
        Y_rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset[::subsample]]).T
        Y_force = np.vstack([p.force for dataset in datasets for p in dataset[::subsample]]).T
        Y_torque = np.vstack([p.torque for dataset in datasets for p in dataset[::subsample]]).T
        if gmm_hyperparameter_tuning:
            # GMM/GMR on the position
            gmm = GaussianMixtureModel(n_demos=H, n_components_range=np.arange(4, 20))
            gmm.fit(X, Y_pos)
            mu_pos, sigma_pos = gmm.predict(x)
            # GMM/GMR on the orientation
            gmm.fit(X, Y_rot)
            mu_rot, sigma_rot = gmm.predict(x)
            qa = datasets[0][0].rot
            quats = np.vstack((mu_rot,np.zeros_like(mu_rot[0,:])))
            for i in range(mu_rot.shape[1]):
                quats[:, i] = (Quaternion.exp(mu_rot[:, i])*qa).as_array()
            # GMM/GMR on the force
            gmm.fit(X, Y_force)
            mu_force, sigma_force = gmm.predict(x)
            # GMM/GMR on the torque
            gmm.fit(X, Y_torque)
            mu_torque, sigma_torque = gmm.predict(x)
        else:
            # GMM/GMR on the position
            gmm = GaussianMixtureModel(n_demos=H, n_components=12)
            gmm.fit(X, Y_pos)
            mu_pos, sigma_pos = gmm.predict(x)
            # GMM/GMR on the orientation
            gmm = GaussianMixtureModel(n_demos=H, n_components=13)
            gmm.fit(X, Y_rot)
            mu_rot, sigma_rot = gmm.predict(x)
            qa = datasets[0][0].rot
            quats = np.vstack((mu_rot,np.zeros_like(mu_rot[0,:])))
            for i in range(mu_rot.shape[1]):
                quats[:, i] = (Quaternion.exp(mu_rot[:, i])*qa).as_array()
            # GMM/GMR on the force
            gmm = GaussianMixtureModel(n_demos=H, n_components=12)
            gmm.fit(X, Y_force)
            mu_force, sigma_force = gmm.predict(x)
            # GMM/GMR on the torque
            gmm = GaussianMixtureModel(n_demos=H, n_components=14)
            gmm.fit(X, Y_torque)
            mu_torque, sigma_torque = gmm.predict(x)
        # Hyperparameter tuning
        if kmp_hyperparameter_tuning:
            position = []
            orientation = []
            force = []
            torque = []
            for l in [0.1, 1, 10, 100, 1000]:
                for lc in [0.1, 1, 10, 100, 1000]:
                    for kernel_gamma in [0.1, 1, 10, 100, 1000]:
                        kmp = KMP(l=l, lc=lc, kernel_gamma=kernel_gamma)
                        # KMP on the position
                        kmp.fit(x, mu_pos, sigma_pos)
                        kmp.predict(x) 
                        position.append([l, lc, kernel_gamma, kmp.kl_divergence])
                        # KMP on the orientation
                        kmp.fit(x, mu_rot, sigma_rot)
                        kmp.predict(x)
                        file_path = join(ROOT, 'hyperparameter_tuning/orientation.csv')
                        orientation.append([l, lc, kernel_gamma, kmp.kl_divergence])
                        # KMP on the force
                        kmp.fit(x, mu_force, sigma_force)
                        kmp.predict(x)
                        force.append([l, lc, kernel_gamma, kmp.kl_divergence])
                        # KMP on the torque
                        kmp.fit(x, mu_torque, sigma_torque)
                        kmp.predict(x)
                        torque.append([l, lc, kernel_gamma, kmp.kl_divergence])
            position = np.array(position)
            orientation = np.array(orientation)
            force = np.array(force)
            torque = np.array(torque)
            # Select the params where KL is the minimum
            position = position[np.argmin(position[:,3]), :3]
            orientation = orientation[np.argmin(orientation[:,3]), :3]
            force = force[np.argmin(force[:,3]), :3]
            torque = torque[np.argmin(torque[:,3]), :3]
        else:
            # Skip parameter tuning
            position = [0.1, 0.1, 1000]
            orientation = [10, 0.1, 1000]
            force = [1000, 0.1, 1000]
            torque = [1, 0.1, 100]
        # Train the KMPs with the best hyperparameters
        # KMP on the position
        kmp = KMP(l=position[0], lc=position[1], kernel_gamma=position[2])
        kmp.fit(x, mu_pos, sigma_pos)
        mu_pos_kmp, sigma_pos_kmp = kmp.predict(x) 
        # KMP on the orientation
        kmp = KMP(l=orientation[0], lc=orientation[1], kernel_gamma=orientation[2])
        kmp.fit(x, mu_rot, sigma_rot)
        mu_rot_kmp, sigma_rot_kmp = kmp.predict(x)
        quats_kmp = np.vstack((mu_rot_kmp,np.zeros_like(mu_rot_kmp[0,:])))
        for i in range(mu_rot.shape[1]):
            quats_kmp[:, i] = (Quaternion.exp(mu_rot_kmp[:, i])*qa).as_array()
        # KMP on the force
        kmp = KMP(l=force[0], lc=force[1], kernel_gamma=force[2])
        kmp.fit(x, mu_force, sigma_force)
        mu_force_kmp, sigma_force_kmp = kmp.predict(x)
        # KMP on the torque
        kmp = KMP(l=torque[0], lc=torque[1], kernel_gamma=torque[2])
        kmp.fit(x, mu_torque, sigma_torque)
        mu_torque_kmp, sigma_torque_kmp = kmp.predict(x)
        # Plot everything
        fig, ax = plt.subplots(4, 3)
        for dataset in datasets:
            self.plot_demo(ax, dataset[::subsample])
        t = x.flatten()
        for i in range(3):
            # GMR
            ax[0, i].plot(t, mu_pos[i, :], color='red', linestyle='dashed')
            ax[1, i].plot(t, quats[i + 1, :], color='red', linestyle='dashed')
            ax[2, i].plot(t, mu_force[i, :], color='red', linestyle='dashed')
            ax[3, i].plot(t, mu_torque[i, :], color='red', linestyle='dashed')
            # KMP
            ax[0, i].plot(t, mu_pos_kmp[i, :], color='green')
            ax[1, i].plot(t, quats_kmp[i + 1, :], color='green')
            ax[2, i].plot(t, mu_force_kmp[i, :], color='green')
            ax[3, i].plot(t, mu_torque_kmp[i, :], color='green')
        plt.show()


    def plot_demo(self, ax, demonstration, linestyle='solid', label=''):
        # Plot all the data in a demonstration
        time = [0.001*i for i in range(len(demonstration))]
        x = [p.x for p in demonstration]
        y = [p.y for p in demonstration]
        z = [p.z for p in demonstration]
        qx = [p.rot.as_array()[1] for p in demonstration]    
        qy = [p.rot.as_array()[2] for p in demonstration]    
        qz = [p.rot.as_array()[3] for p in demonstration]    
        fx = [p.fx for p in demonstration]    
        fy = [p.fy for p in demonstration]    
        fz = [p.fz for p in demonstration]
        tx = [p.mx for p in demonstration]
        ty = [p.my for p in demonstration]
        tz = [p.mz for p in demonstration]
        data = [x, y, z, qx, qy, qz, fx, fy, fz, tx, ty, tz]
        y_labels = ['x [m]', 'y [m]', 'z [m]',
                    'qx', 'qy', 'qz',
                    'Fx [N]', 'Fy [N]', 'Fz [N]',
                    'Mx - [Nmm]', 'My - [Nmm]', 'Mz - [Nmm]']
        for i in range(4):
            for j in range(3):
                ax[i, j].plot(time, data[i*3 + j],
                            linestyle=linestyle, label=label, linewidth=0.6, color='grey')
                ax[i, j].set_ylabel(y_labels[i*3 + j])
                if i == 3:
                    ax[i, j].set_xlabel('Time [s]')


def main(task):
    if task == '1':
        single_point_task()
    input()


if __name__ == '__main__':
    task = sys.argv[-1] if len(sys.argv) > 1 else None
    main(task)
