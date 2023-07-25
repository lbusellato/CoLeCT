import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle

from os.path import dirname, abspath, join
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
subsample = 100
pos_ = True
rot_ = True
force_ = True

def main():
    # Showcase GMR and KMP on the single point task dataset
    # Load the demonstrations
    datasets = load_datasets('demonstrations/single_point_task', regex=r'tdataset(\d{2})\.npy')
    # Prepare the data for GMM/GMR
    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0]) // subsample  # Length of each demonstration
    gmm_dt = 0.1
    demo_dura = gmm_dt * (N + 1) # Duration of each demonstration
    x_gmr = gmm_dt*np.arange(1, N + 2).reshape(1, -1)
    X = np.tile(x_gmr, H).reshape(1, -1)
    # Prepare data for GMM/GMR
    Y_pos = np.vstack([p.position for dataset in datasets for p in dataset[::subsample]]).T
    # Compute (approximated) linear velocities
    positions = np.split(copy.deepcopy(Y_pos), H, axis=1)
    for position in positions:
        position[0,:] = np.gradient(position[0,:])/gmm_dt
        position[1,:] = np.gradient(position[1,:])/gmm_dt
        position[2,:] = np.gradient(position[2,:])/gmm_dt
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
    # Recover the auxiliary quaternion  
    qa = datasets[0][0].rot
    # GMM/GMR on the position
    if pos_:
        gmm = GaussianMixtureModel(n_components=10)
        gmm.fit(X_pos)
        mu_pos, sigma_pos = gmm.predict(x_gmr)
        np.save(join(ROOT, 'trained_models/mu_pos.npy'), mu_pos)
        np.save(join(ROOT, 'trained_models/sigma_pos.npy'), sigma_pos)
    # GMM/GMR on the orientation
    if rot_:
        gmm = GaussianMixtureModel(n_components=10)
        gmm.fit(X_rot)
        mu_rot, sigma_rot = gmm.predict(x_gmr)
        np.save(join(ROOT, 'trained_models/mu_rot.npy'), mu_rot)
        np.save(join(ROOT, 'trained_models/sigma_rot.npy'), sigma_rot)
        quats = np.vstack((mu_rot[:3,:],np.zeros_like(mu_rot[0,:])))
        for i in range(mu_rot.shape[1]):
            quats[:, i] = (Quaternion.exp(mu_rot[:3, i])*~qa).as_array()
    Y_force = np.vstack([p.force for dataset in datasets for p in dataset[::subsample]]).T
    # Compute the derivatives of the forces
    forces = np.split(copy.deepcopy(Y_force), H, axis=1)
    for force in forces:
        force[0,:] = np.gradient(force[0,:])/gmm_dt
        force[1,:] = np.gradient(force[1,:])/gmm_dt
        force[2,:] = np.gradient(force[2,:])/gmm_dt
    dY_force = np.hstack(forces)
    poses = np.vstack((Y_pos[:3, :], Y_rot[:3, :]))
    X_force = np.vstack((poses, Y_force, dY_force))
    # GMM/GMR on the force
    if force_:
        gmm = GaussianMixtureModel(n_components=10)
        gmm.fit(X_force)
        mu_force, sigma_force = gmm.predict(x_gmr)
        np.save(join(ROOT, 'trained_models/mu_force.npy'), mu_force)
        np.save(join(ROOT, 'trained_models/sigma_force.npy'), sigma_force)
    # KMP on the position
    if pos_:
        kmp_dt = 0.002
        x_kmp = np.arange(kmp_dt, demo_dura, kmp_dt).reshape(1, -1)
        kmp = KMP(l=0.5, alpha=40, sigma_f=1, verbose=True)
        kmp.fit(x_gmr, mu_pos, sigma_pos)
        mu_pos_kmp, sigma_pos_kmp = kmp.predict(x_kmp) 
        np.save(join(ROOT, 'trained_models/mu_pos_kmp.npy'), mu_pos_kmp)
        np.save(join(ROOT, 'trained_models/sigma_pos_kmp.npy'), sigma_pos_kmp)
        with open(join(ROOT, "trained_models/pos_kmp.pkl"), "wb") as file:
            pickle.dump(kmp, file)
    # KMP on the orientation
    if rot_:
        kmp_dt = 0.002
        x_kmp = np.arange(kmp_dt, demo_dura, kmp_dt).reshape(1, -1)
        kmp = KMP(l=0.5, alpha=30, sigma_f=2, verbose=True)
        kmp.fit(x_gmr, mu_rot, sigma_rot)
        mu_rot_kmp, sigma_rot_kmp = kmp.predict(x_kmp)
        np.save(join(ROOT, 'trained_models/mu_rot_kmp.npy'), mu_rot_kmp)
        np.save(join(ROOT, 'trained_models/sigma_rot_kmp.npy'), sigma_rot_kmp)
        quats_kmp = np.vstack((mu_rot_kmp[:3,:],np.zeros_like(mu_rot_kmp[0,:])))
        for i in range(mu_rot_kmp.shape[1]):
            quats_kmp[:, i] = (Quaternion.exp(mu_rot_kmp[:3, i])*~qa).as_array()
        np.save(join(ROOT, 'trained_models/mu_rot_kmp_quats.npy'), quats_kmp)
        with open(join(ROOT, "trained_models/rot_kmp.pkl"), "wb") as file:
            pickle.dump(kmp, file)
    # KMP on the force
    if force_:
        kmp_dt = 0.002
        x_kmp = np.arange(kmp_dt, demo_dura, kmp_dt).reshape(1, -1)
        kmp = KMP(l=0.5, alpha=40, sigma_f=1, verbose=True)
        kmp.fit(x_gmr, mu_force, sigma_force)
        mu_force_kmp, sigma_force_kmp = kmp.predict(x_kmp)
        np.save(join(ROOT, 'trained_models/mu_force_kmp.npy'), mu_force_kmp)
        np.save(join(ROOT, 'trained_models/sigma_force_kmp.npy'), sigma_force_kmp)
        with open(join(ROOT, "trained_models/force_kmp.pkl"), "wb") as file:
            pickle.dump(kmp, file)
    # Plot everything
    fig, ax = plt.subplots(2, 3, figsize=(16,8))
    for dataset in datasets:
        plot_demo(ax, dataset[::subsample], demo_dura)
    t_gmr = np.arange(gmm_dt, demo_dura, gmm_dt)
    t_kmp = np.arange(kmp_dt, demo_dura, kmp_dt)
    for i in range(3):
        if pos_:
            ax[0, i].errorbar(x=t_gmr, y=mu_pos[i, :], yerr=np.sqrt(sigma_pos[i,i,:]), color='red', alpha=0.35)
            ax[0, i].errorbar(x=t_kmp, y=mu_pos_kmp[i, :], yerr=np.sqrt(sigma_pos_kmp[i,i,:]), color='green', alpha=0.25)
        if rot_:            
            """# Plot Euclidean quaternions
            ax[1, i].errorbar(x=t_gmr, y=mu_rot[i, :], yerr=np.sqrt(sigma_rot[i,i,:]), color='red', alpha=0.35)
            ax[1, i].errorbar(x=t_kmp, y=mu_rot_kmp[i, :], yerr=np.sqrt(sigma_rot_kmp[i,i,:]), color='green', alpha=0.1)
            ax[1, i].plot(t_gmr, mu_rot[i, :], color='green')
            ax[1, i].plot(t_kmp, mu_rot_kmp[i, :], color='green')"""
            ax[1, i].plot(t_gmr, quats[i+1, :], color='red')
            ax[1, i].plot(t_kmp, quats_kmp[i+1, :], color='green')
    fig.suptitle('Single point task - GMR and KMP')
    fig.tight_layout()
    plots_path = join(ROOT, 'media/single_point_task_kmp.png')
    plt.savefig(plots_path)
    plt.show()

def plot_demo(ax, demonstration, dura, linestyle='solid', label=''):
    # Plot all the data in a demonstration
    time = np.arange(0.1, dura, 0.1)
    x = [p.x for p in demonstration]
    y = [p.y for p in demonstration]
    z = [p.z for p in demonstration]
    qx = [p.rot.as_array()[1] for p in demonstration]    
    qy = [p.rot.as_array()[2] for p in demonstration]    
    qz = [p.rot.as_array()[3] for p in demonstration]  
    data = [x, y, z, qx, qy, qz, fx, fy, fz]
    y_labels = ['x [m]', 'y [m]', 'z [m]',
                'qx', 'qy', 'qz']
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(time, data[i*3 + j],
                        linestyle=linestyle, label=label, linewidth=0.6, color='grey')
            ax[i, j].set_ylabel(y_labels[i*3 + j])
            ax[i, j].grid(True)
            if i == 1:
                ax[i, j].set_xlabel('Time [s]')


if __name__ == '__main__':
    main()
