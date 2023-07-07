import logging
import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import dirname, abspath, join, exists
from src.dataset import load_datasets
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
subsample = 25

def main():
    # Showcase GMR and KMP on the single point task dataset
    # Load the demonstrations
    datasets = load_datasets('demonstrations/single_point_task')
    # Prepare the data for GMM/GMR
    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0]) // subsample  # Length of each demonstration
    dt = 0.1
    x = dt*np.arange(1, N + 1).reshape(1, -1)
    X = dt*np.tile(np.arange(N + 1), H).reshape(1, -1)
    Y_pos = np.vstack([p.position for dataset in datasets for p in dataset[::subsample]]).T
    Y_rot = np.vstack([p.rot_eucl for dataset in datasets for p in dataset[::subsample]]).T
    Y_force = np.vstack([p.force for dataset in datasets for p in dataset[::subsample]]).T
    Y_torque = np.vstack([p.torque for dataset in datasets for p in dataset[::subsample]]).T
    qa = datasets[0][0].rot
    # GMM/GMR hyperparameter tuning
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
        if True or not exists(join(ROOT, 'trained_models/mu_pos.npy')):
            # GMM/GMR on the position
            gmm = GaussianMixtureModel(n_demos=H, n_components=12)
            gmm.fit(X, Y_pos)
            mu_pos, sigma_pos = gmm.predict(x)
            np.save(join(ROOT, 'trained_models/mu_pos.npy'), mu_pos)
            np.save(join(ROOT, 'trained_models/sigma_pos.npy'), sigma_pos)
            # GMM/GMR on the orientation
            gmm = GaussianMixtureModel(n_demos=H, n_components=13)
            gmm.fit(X, Y_rot)
            mu_rot, sigma_rot = gmm.predict(x)
            np.save(join(ROOT, 'trained_models/mu_rot.npy'), mu_rot)
            np.save(join(ROOT, 'trained_models/sigma_rot.npy'), sigma_rot)
            quats = np.vstack((mu_rot,np.zeros_like(mu_rot[0,:])))
            for i in range(mu_rot.shape[1]):
                quats[:, i] = (Quaternion.exp(mu_rot[:, i])*qa).as_array()
            # GMM/GMR on the force
            gmm = GaussianMixtureModel(n_demos=H, n_components=12)
            gmm.fit(X, Y_force)
            mu_force, sigma_force = gmm.predict(x)
            np.save(join(ROOT, 'trained_models/mu_force.npy'), mu_force)
            np.save(join(ROOT, 'trained_models/sigma_force.npy'), sigma_force)
            # GMM/GMR on the torque
            gmm = GaussianMixtureModel(n_demos=H, n_components=14)
            gmm.fit(X, Y_torque)
            mu_torque, sigma_torque = gmm.predict(x)
            np.save(join(ROOT, 'trained_models/mu_torque.npy'), mu_torque)
            np.save(join(ROOT, 'trained_models/sigma_torque.npy'), sigma_torque)
        else:
            mu_pos = np.load(join(ROOT, 'trained_models/mu_pos.npy'))
            sigma_pos = np.load(join(ROOT, 'trained_models/sigma_pos.npy'))
            mu_rot = np.load(join(ROOT, 'trained_models/mu_rot.npy'))
            quats = np.vstack((mu_rot,np.zeros_like(mu_rot[0,:])))
            for i in range(mu_rot.shape[1]):
                quats[:, i] = (Quaternion.exp(mu_rot[:, i])*qa).as_array()
            sigma_rot = np.load(join(ROOT, 'trained_models/sigma_rot.npy'))
            mu_force = np.load(join(ROOT, 'trained_models/mu_force.npy'))
            sigma_force = np.load(join(ROOT, 'trained_models/sigma_force.npy'))
            mu_torque = np.load(join(ROOT, 'trained_models/mu_torque.npy'))
            sigma_torque = np.load(join(ROOT, 'trained_models/sigma_torque.npy'))
    # KMP hyperparameter tuning
    if kmp_hyperparameter_tuning:
        position = []
        orientation = []
        force = []
        torque = []
        for lambda1 in [0.01, 0.1, 1, 10, 100]:
            for lambda2 in [0.01, 0.1, 1, 10, 100]:
                for l in [0.01, 0.1, 1, 10, 100]:
                    for sigma_f in [0.01, 0.1, 1, 10, 100]:
                        kmp = KMP(lambda1=lambda1, 
                                    lambda2=lambda2, 
                                    l=l,
                                    sigma_f=sigma_f)
                        # KMP on the position
                        kmp.fit(x, mu_pos, sigma_pos)
                        kmp.predict(x) 
                        position.append([lambda1, lambda2, l, sigma_f, kmp.kl_divergence])
                        # KMP on the orientation
                        kmp.fit(x, mu_rot, sigma_rot)
                        kmp.predict(x)
                        orientation.append([lambda1, lambda2, l, sigma_f, kmp.kl_divergence])
                        # KMP on the force
                        kmp.fit(x, mu_force, sigma_force)
                        kmp.predict(x)
                        force.append([lambda1, lambda2, l, sigma_f, kmp.kl_divergence])
                        # KMP on the torque
                        kmp.fit(x, mu_torque, sigma_torque)
                        kmp.predict(x)
                        torque.append([lambda1, lambda2, l, sigma_f, kmp.kl_divergence])
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
        position = [0.1, 0.1, 0.001, 1]
        orientation = [10, 0.1, 0.001, 1]
        force = [0.1, 1, 0.01, 1]
        torque = [0.1, 1, 0.01, 1]
    # Train the KMPs with the best hyperparameters
    if True or not exists(join(ROOT, 'trained_models/mu_pos_kmp.npy')):
        # KMP on the position
        kmp = KMP(lambda1=position[0], 
                lambda2=position[1], 
                l=position[2], 
                sigma_f=position[3],
                verbose=True)
        kmp.fit(x, mu_pos, sigma_pos)
        mu_pos_kmp, sigma_pos_kmp = kmp.predict(x) 
        np.save(join(ROOT, 'trained_models/mu_pos_kmp.npy'), mu_pos_kmp)
        np.save(join(ROOT, 'trained_models/sigma_pos_kmp.npy'), sigma_pos_kmp)
        # KMP on the orientation
        kmp = KMP(lambda1=orientation[0], 
                lambda2=orientation[1], 
                l=orientation[2], 
                sigma_f=orientation[3],
                verbose=True)
        kmp.fit(x, mu_rot, sigma_rot)
        mu_rot_kmp, sigma_rot_kmp = kmp.predict(x)
        np.save(join(ROOT, 'trained_models/mu_rot_kmp.npy'), mu_rot_kmp)
        np.save(join(ROOT, 'trained_models/sigma_rot_kmp.npy'), sigma_rot_kmp)
        quats_kmp = np.vstack((mu_rot_kmp,np.zeros_like(mu_rot_kmp[0,:])))
        for i in range(mu_rot.shape[1]):
            quats_kmp[:, i] = (Quaternion.exp(mu_rot_kmp[:, i])*qa).as_array()
        # KMP on the force
        kmp = KMP(lambda1=force[0], 
                lambda2=force[1], 
                l=force[2], 
                sigma_f=force[3],
                verbose=True)
        kmp.fit(x, mu_force, sigma_force)
        mu_force_kmp, sigma_force_kmp = kmp.predict(x)
        np.save(join(ROOT, 'trained_models/mu_force_kmp.npy'), mu_force_kmp)
        np.save(join(ROOT, 'trained_models/sigma_force_kmp.npy'), sigma_force_kmp)
        # KMP on the torque
        kmp = KMP(lambda1=torque[0], 
                lambda2=torque[1], 
                l=torque[2], 
                sigma_f=torque[3],
                verbose=True)
        kmp.fit(x, mu_torque, sigma_torque)
        mu_torque_kmp, sigma_torque_kmp = kmp.predict(x)
        np.save(join(ROOT, 'trained_models/mu_torque_kmp.npy'), mu_torque_kmp)
        np.save(join(ROOT, 'trained_models/sigma_torque_kmp.npy'), sigma_torque_kmp)
    else:
        mu_pos_kmp = np.load(join(ROOT, 'trained_models/mu_pos_kmp.npy'))
        sigma_pos_kmp = np.load(join(ROOT, 'trained_models/sigma_pos_kmp.npy'))
        mu_rot_kmp = np.load(join(ROOT, 'trained_models/mu_rot_kmp.npy'))
        quats_kmp = np.vstack((mu_rot_kmp,np.zeros_like(mu_rot_kmp[0,:])))
        for i in range(mu_rot.shape[1]):
            quats_kmp[:, i] = (Quaternion.exp(mu_rot_kmp[:, i])*qa).as_array()
        sigma_rot_kmp = np.load(join(ROOT, 'trained_models/sigma_rot_kmp.npy'))
        mu_force_kmp = np.load(join(ROOT, 'trained_models/mu_force_kmp.npy'))
        sigma_force_kmp = np.load(join(ROOT, 'trained_models/sigma_force_kmp.npy'))
        mu_torque_kmp = np.load(join(ROOT, 'trained_models/mu_torque_kmp.npy'))
        sigma_torque_kmp = np.load(join(ROOT, 'trained_models/sigma_torque_kmp.npy'))
    # Plot everything
    fig, ax = plt.subplots(4, 3, figsize=(16,8))
    for dataset in datasets:
        plot_demo(ax, dataset[::subsample])
    t = x.flatten()
    for i in range(3):
        # GMR
        ax[0, i].plot(t, mu_pos[i, :], color='red', linestyle='dashed')
        ax[1, i].plot(t, quats[i + 1, :], color='red', linestyle='dashed')
        ax[2, i].plot(t, mu_force[i, :], color='red', linestyle='dashed')
        ax[3, i].plot(t, mu_torque[i, :], color='red', linestyle='dashed')
        # KMP
        ax[0, i].errorbar(x=t, y=mu_pos_kmp[i, :], yerr=sigma_pos_kmp[i,i,:], color='yellow', alpha=0.25)
        ax[0, i].plot(t, mu_pos_kmp[i, :], color='green')
        ax[1, i].plot(t, quats_kmp[i + 1, :], color='green')
        ax[2, i].errorbar(x=t, y=mu_force_kmp[i, :], yerr=sigma_force_kmp[i,i,:], color='yellow', alpha=0.25)
        ax[3, i].errorbar(x=t, y=mu_torque_kmp[i, :], yerr=sigma_torque_kmp[i,i,:], color='yellow', alpha=0.25)
        ax[2, i].plot(t, mu_force_kmp[i, :], color='green')
        ax[3, i].plot(t, mu_torque_kmp[i, :], color='green')
    fig.suptitle('Single point task - GMR and KMP')
    fig.tight_layout()
    plots_path = join(ROOT, 'media/single_point_task_kmp.png')
    plt.savefig(plots_path)
    plt.show()

def plot_demo(ax, demonstration, linestyle='solid', label=''):
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
            ax[i, j].grid(True)
            if i == 3:
                ax[i, j].set_xlabel('Time [s]')


if __name__ == '__main__':
    main()
