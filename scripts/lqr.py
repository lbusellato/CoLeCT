import logging
import matplotlib.pyplot as plt
import numpy as np

from control import lqr
from os.path import dirname, abspath, join, exists

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
    # Showcase the usage of LQR to compute controller gains
    # Recover the uncertainties (i.e. the covariance matrices)
    sigma_force_kmp = np.load(join(ROOT, 'trained_models/sigma_force_kmp.npy'))
    sigma_torque_kmp = np.load(join(ROOT, 'trained_models/sigma_torque_kmp.npy'))
    # Compute covariance matrix inverses
    if not exists(join(ROOT, 'trained_models/sigma_force_kmp_inv.npy')):
        sigma_force_kmp_inv = np.zeros_like(sigma_force_kmp)
        for i in range(sigma_force_kmp.shape[2]):
            sigma_force_kmp_inv[:,:,i] = np.matrix.round(np.linalg.inv(sigma_force_kmp[:,:,i]), 2)
        np.save(join(ROOT, 'trained_models/sigma_force_kmp_inv.npy'), sigma_force_kmp_inv)
        sigma_torque_kmp_inv = np.zeros_like(sigma_torque_kmp)
        for i in range(sigma_torque_kmp.shape[2]):
            sigma_torque_kmp_inv[:,:,i] = np.matrix.round(np.linalg.inv(sigma_torque_kmp[:,:,i]), 2)
        np.save(join(ROOT, 'trained_models/sigma_torque_kmp_inv.npy'), sigma_torque_kmp_inv)
    else:
        sigma_force_kmp_inv = np.load(join(ROOT, 'trained_models/sigma_force_kmp_inv.npy'))
        sigma_torque_kmp_inv = np.load(join(ROOT, 'trained_models/sigma_torque_kmp_inv.npy'))
    # Compute the gains for the force/torque controller using LQR
    A = np.eye(3)
    B = np.eye(3)
    gain_magnitude_penalty = 0.1
    R = np.eye(3)*gain_magnitude_penalty
    # Forces
    Ks_force = []
    sigmas_force = []
    for i in range(sigma_force_kmp_inv.shape[2]):
        Q = sigma_force_kmp_inv[:,:,i]
        K, S, E = lqr(A, B, Q, R)
        Ks_force.append([K[0,0], K[1,1], K[2,2]])
        sigma = sigma_force_kmp[:,:,i]
        sigmas_force.append([sigma[0,0], sigma[1,1], sigma[2,2]])
    Ks_force = np.array(Ks_force)
    sigmas_force = np.array(sigmas_force)
    # Torques
    Ks_torque = []
    sigmas_torque = []
    for i in range(sigma_torque_kmp_inv.shape[2]):
        Q = sigma_torque_kmp_inv[:,:,i]
        K, S, E = lqr(A, B, Q, R)
        Ks_torque.append([K[0,0], K[1,1], K[2,2]])
        sigma = sigma_torque_kmp[:,:,i]
        sigmas_torque.append([sigma[0,0], sigma[1,1], sigma[2,2]])
    Ks_torque = np.array(Ks_torque)
    sigmas_torque = np.array(sigmas_torque)
    # Plot everything
    fig, ax = plt.subplots(2,3, figsize=(16,8))
    t = 0.001*np.arange(sigma_force_kmp_inv.shape[2])
    ax[0,0].plot(t, Ks_force[:,0], color='blue')
    tmp = ax[0,0].twinx()
    tmp.plot(t, sigmas_force[:,0], color='red')
    tmp.set_ylabel('$\sigma^2 F_x$', color='red')
    ax[0,1].plot(t, Ks_force[:,1], color='blue')
    tmp = ax[0,1].twinx()
    tmp.plot(t, sigmas_force[:,1], color='red')
    tmp.set_ylabel('$\sigma^2 F_y$', color='red')
    ax[0,2].plot(t, Ks_force[:,2], color='blue')
    tmp = ax[0,2].twinx()
    tmp.plot(t, sigmas_force[:,2], color='red')
    tmp.set_ylabel('$\sigma^2 F_z$', color='red')
    ax[1,0].plot(t, Ks_torque[:,0], color='blue')
    tmp = ax[1,0].twinx()
    tmp.plot(t, sigmas_torque[:,0], color='red')
    tmp.set_ylabel('$\sigma^2 M_x$', color='red')
    ax[1,1].plot(t, Ks_torque[:,1], color='blue')
    tmp = ax[1,1].twinx()
    tmp.plot(t, sigmas_torque[:,1], color='red')
    tmp.set_ylabel('$\sigma^2 M_y$', color='red')
    ax[1,2].plot(t, Ks_torque[:,2], color='blue')
    tmp = ax[1,2].twinx()
    tmp.plot(t, sigmas_torque[:,2], color='red')
    tmp.set_ylabel('$\sigma^2 M_z$', color='red')
    ax[0,0].set_ylabel('$K_{F_x}$', color='blue')
    ax[0,1].set_ylabel('$K_{F_y}$', color='blue')
    ax[0,2].set_ylabel('$K_{F_z}$', color='blue')
    ax[1,0].set_ylabel('$K_{M_x}$', color='blue')
    ax[1,1].set_ylabel('$K_{M_y}$', color='blue')
    ax[1,2].set_ylabel('$K_{M_z}$', color='blue')
    for i in range(2):
        for j in range(3):
            if i == 1:
                ax[i,j].set_xlabel('Time [s]')
            ax[i,j].grid()
    fig.suptitle('LQR controller gains vs KMP covariances')
    fig.tight_layout()
    plots_path = join(ROOT, 'media/single_point_task_lqr.png')
    plt.savefig(plots_path)
    plt.show()


if __name__ == '__main__':
    main()
