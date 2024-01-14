import copy
import logging
import numpy as np

from numpy.linalg import inv, norm
from scipy.stats import multivariate_normal
from typing import Tuple

REALMIN = np.finfo(np.float64).tiny  # To avoid division by 0

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)


class KMP:
    """Trajectory imitation and adaptation using Kernelized Movement Primitives.

    Parameters
    ----------
    l : int, default=0.5
        Lambda regularization factor for the minimization problem.
    alpha : float, default=40
        Coefficient for the covariance prediction.
    sigma_f : float, default=1
        Kernel coefficient.
    tol : float, default=0.0005
        Tolerance for the discrimination of conflicting points.
    time_driven_kernel : bool, default=True
        Enable/disable the use of the time-driven kernel. Realistically, should be turned off
        only in the position-only input demo.
    verbose : bool, default=True
        Enable/disable verbose output.
    """

    def __init__(
        self,
        l: float = 0.5,
        alpha: float = 40,
        sigma_f: float = 1.0,
        adaptation_tol: float = 0.0005,
        time_driven_kernel: bool = True,
        verbose: bool = False,
    ) -> None:
        self.l = l
        self.alpha = alpha
        self.sigma_f = sigma_f
        self.kl_divergence = None
        self.tol = adaptation_tol
        self.time_driven_kernel = time_driven_kernel
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)

    def set_waypoint(self, 
                     s: np.ndarray, 
                     xi: np.ndarray, 
                     sigma: np.ndarray) -> None:
        """Adds a waypoint to the database, checking for conflicts.

        Parameters
        ----------
        s : np.ndarray of shape (n_input_features,n_samples)
            Array of input vectors.
        xi : np.ndarray of shape (n_output_features,n_samples)
            Array of output vectors
        sigma : np.ndarray of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        for j in range(len(s)):
            # Loop over the reference database to find any conflicts
            min_dist = np.inf
            for i in range(self.N):
                dist = np.linalg.norm(self.s[:,i]-s[j])
                if  dist < min_dist:
                    min_dist = dist
                    id = i
            if min_dist < self.tol:
                # Replace the conflicting point
                self.s[:,id] = s[j]
                self.xi[:,id] = xi[j]
                self.sigma[:,:,id] = sigma[j]
            else:
                # Add the new point to the database
                self.s = np.concatenate((self.s, np.array(s[j]).reshape(1,-1)),axis=1)
                self.xi = np.concatenate((self.xi, xi[j].T),axis=1)
                self.sigma = np.concatenate((self.sigma, np.expand_dims(sigma[j],2)),axis=2)
        # Refit the model with the new data
        self.fit(self.s, self.xi, self.sigma)

    def __kernel_matrix(self, t1: float, t2: float) -> np.ndarray:
        """Computes the kernel matrix for the given input pair.

        Parameters
        ----------
        t1 : float
            The first input.
        t2 : float
            The second input.

        Returns
        -------
        kernel : np.ndarray of shape (n_features,n_features)
            The kernel matrix evaluated in the provided input pair.
        """

        def kernel(s1, s2):
            return np.exp(-self.sigma_f * norm(s1 - s2) ** 2)

        # Compute the kernels
        dt = 0.001
        ktt = kernel(t1, t2)
        if not self.time_driven_kernel:
            return ktt*np.eye(self.O)
        ktdt_tmp = kernel(t1, t2 + dt)
        kdtt_tmp = kernel(t1 + dt, t2)
        kdtdt_tmp = kernel(t1 + dt, t2 + dt)
        # Components of the matrix
        ktdt = (ktdt_tmp - ktt) / dt
        kdtt = (kdtt_tmp - ktt) / dt
        kdtdt = (kdtdt_tmp - ktdt_tmp - kdtt_tmp + ktt) / dt**2
        # Fill the kernel matrix
        O = self.O // 2
        kernel_matrix = np.block([[ktt*np.eye(O), ktdt*np.eye(O)],[kdtt*np.eye(O), kdtdt*np.eye(O)]])

        return kernel_matrix

    def fit(self, X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> None:
        """Train the model by computing the estimator matrix inv(K+lambda*sigma).

        Parameters
        ----------
        X : np.ndarray of shape (n_input_features,n_samples)
            Array of input vectors.
        mu : np.ndarray of shape (n_output_features,n_samples)
            Array of output vectors
        var : np.ndarray of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        self.s = copy.deepcopy(X)
        self.xi = copy.deepcopy(mu)
        self.sigma = copy.deepcopy(var)
        self.O, self.N = self.xi.shape
        k = np.zeros((self.N * self.O, self.N * self.O))
        # Construct the estimator
        for i in range(self.N):
            for j in range(self.N):
                kernel = self.__kernel_matrix(self.s[:, i], self.s[:, j])
                k[i * self.O : (i + 1) * self.O, j * self.O : (j + 1) * self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    k[j * self.O : (j + 1) * self.O, i * self.O : (i + 1) * self.O] += (
                        self.l * self.sigma[:, :, i]
                    )
        self._estimator = inv(k)
        self._logger.info("KMP fit done.")

    def predict(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Carry out a prediction on the mean and covariance associated to the given input.

        Parameters
        ----------
        s : np.ndarray of shape (n_features,n_samples)
            The set of inputs to make a prediction of.

        Returns
        -------
        xi : np.ndarray of shape (n_features,n_samples)
            The array of predicted means.

        sigma : np.ndarray of shape (n_features,n_features,n_samples)
            The array of predicted covariance matrices.
        """
        # If a single point is queried, make sure it is in the proper shape
        if s.shape == (s.size,):
            s = s.reshape((s.size, 1))
        xi = np.zeros((self.O, s.shape[1]))
        sigma = np.zeros((self.O, self.O, s.shape[1]))
        for j in range(s.shape[1]):
            k = np.zeros((self.O, self.N * self.O))
            Y = np.zeros(self.N * self.O)
            for i in range(self.N):
                k[:, i * self.O : (i + 1) * self.O] = self.__kernel_matrix(
                    s[:, j], self.s[:, i]
                )
                for h in range(self.O):
                    Y[i * self.O + h] = self.xi[h, i]
            xi[:, j] = k @ self._estimator @ Y
            sigma[:, :, j] = self.alpha * (
                self.__kernel_matrix(s[:, j], s[:, j]) - k @ self._estimator @ k.T
            )
        self._logger.info("KMP predict done.")
        #self.kl_divergence = self.KL_divergence(xi, sigma, self.xi, self.sigma)
        self.kl_divergence = self.mean_kl_divergence(xi, sigma, self.xi, self.sigma)

        return xi, sigma

    def multivariate_kl_divergence(self, mu1, cov1, mu2, cov2):
        """
        Calculate the KL divergence between two multivariate Gaussian distributions.
        
        Parameters:
        mu1, mu2: means of the distributions
        cov1, cov2: covariance matrices of the distributions
        
        Returns:
        kl_divergence: KL divergence value
        """
        # Ensure the covariance matrices are symmetric
        cov1 = 0.5 * (cov1 + cov1.T)
        cov2 = 0.5 * (cov2 + cov2.T)
        
        # Calculate determinants and inverse matrices
        det_cov1 = max(np.linalg.det(cov1), 1e-16)
        det_cov2 = max(np.linalg.det(cov2), 1e-16)
        inv_cov2 = inv(cov2)

        # Calculate the trace term
        trace_term = np.trace(inv_cov2 @ cov1)

        # Calculate the difference in means
        mean_diff = mu2 - mu1

        # Calculate the KL divergence
        kl_divergence = 0.5 * (np.log(det_cov2 / det_cov1) - mu1.size + trace_term + mean_diff.T @ inv_cov2 @ mean_diff)

        return kl_divergence
    
    def mean_kl_divergence(self, trajectory1_means, trajectory1_covariances, trajectory2_means, trajectory2_covariances):
        """
            Calculate the mean KL divergence between corresponding points of two trajectories.

            Parameters:
            trajectory1_means, trajectory1_covariances: means and covariance matrices of the first trajectory
            trajectory2_means, trajectory2_covariances: means and covariance matrices of the second trajectory
            
            Returns:
            mean_kl_divergence: mean KL divergence value
        """
        num_points = trajectory1_means.shape[1]
        kl_divergences = np.zeros(num_points)

        for i in range(num_points):
            kl_divergences[i] = self.multivariate_kl_divergence(
                trajectory1_means[:,i], trajectory1_covariances[:,:,i],
                trajectory2_means[:,i], trajectory2_covariances[:,:,i]
            )

        mean_kl_divergence = np.mean(kl_divergences)
        return mean_kl_divergence