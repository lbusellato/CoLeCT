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
        # Hyperparameters
        self.l = l
        self.alpha = alpha
        self.sigma_f = sigma_f
        # KL divergence between training and prediction
        self.kl_divergence = None
        # Tolerance for discriminating between conflicting points
        self.tol = adaptation_tol
        
        self.time_driven_kernel = time_driven_kernel
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)

    @property
    def verbose(self) -> bool:
        """
        Get the value of verbose.

        Returns:
            bool: The value of verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """
        Set the value of verbose.

        Args:
            value (bool): The new value for verbose.
        """
        self._verbose = value
        self._logger.setLevel(level=logging.DEBUG if self._verbose else logging.INFO)

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
        # If a single point is queried, make sure it is in the proper shape
        if s.shape == (s.size,):
            s = s.reshape((s.size, 1))
            xi = xi.reshape((xi.size, 1))
            sigma = sigma.reshape((sigma.shape[0], sigma.shape[1], 1))
        for j in range(s.shape[1]):
            # Loop over the reference database to find any conflicts
            min_dist = np.inf
            for i in range(self.N):
                dist = np.linalg.norm(self.s[:,i]-s[:,j])
                if  dist < min_dist:
                    min_dist = dist
                    id = i
            if min_dist < self.tol:
                # Replace the conflicting point
                self.s[:,id] = s[:, j]
                self.xi[:,id] = xi[:, j]
                self.sigma[:,:,id] = sigma[:, :, j]
            else:
                # Add the new point to the database
                self.s = np.insert(self.s, self.s.shape[1], s[:,j], axis=1)
                self.xi = np.insert(self.xi, self.xi.shape[1], xi[:, j].T, axis=1)
                self.sigma = np.insert(self.sigma, self.sigma.shape[2], np.expand_dims(sigma[:, :, j],2), axis=2)
        # Refit the model with the new data
        self.fit(self.s, self.xi, self.sigma)

    def __kernel_matrix(self, t1, t2) -> np.ndarray:
        """Computes the kernel matrices for the given input arrays.

        Parameters
        ----------
        t1_array : np.ndarray or float
            Array of first inputs.
        t2_array : np.ndarray or float
            Array of second inputs.

        Returns
        -------
        kernel_matrices : np.ndarray of shape (n_samples, n_features, n_features)
            The kernel matrices evaluated for the provided input arrays.
        """


        def kernel(s1, s2):
            return np.exp(-self.sigma_f * norm(s1 - s2, axis=0) ** 2)

        # Compute the kernels
        dt = 0.001
        ktt = kernel(t1, t2)
        if not self.time_driven_kernel:
            if isinstance(ktt, np.ndarray):
                return [ktt_*np.eye(self.O) for ktt_ in ktt]
            else:
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
        if isinstance(ktt, np.ndarray):
            kernel_matrix = [np.block([[k1*np.eye(O), k2*np.eye(O)],[k3*np.eye(O), k4*np.eye(O)]]) for k1, k2, k3, k4 in zip(ktt, ktdt, kdtt, kdtdt)]
        else:
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
        N = self.s.shape[1]
        for i in range(N):
            for j in range(N):
                kernel = self.__kernel_matrix(self.s[:, i], self.s[:, j])
                k[i * self.O : (i + 1) * self.O, j * self.O : (j + 1) * self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    k[j * self.O : (j + 1) * self.O, i * self.O : (i + 1) * self.O] += (
                        self.l * self.sigma[:, :, i]
                    )
        self._estimator = np.linalg.pinv(k)
        # In the original MATLAB code Y was computed inside the predict loop, however that's really
        # inefficient. I do the computation here, to speed up the prediction step.
        self.Y = self.xi.T.flatten()
        self._mean_estimator = self.Y @ self._estimator
        self._logger.debug("KMP fit done.")

    def predict(self, s: np.ndarray, compute_KL: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """Carry out a prediction on the mean and covariance associated to the given input.

        Parameters
        ----------
        s : np.ndarray of shape (n_features,n_samples)
            The set of inputs to make a prediction of.
        compute_KL : bool
            Whether or not to compute the KL divergence. Should be True only while tuning the parameters (but not always even then).

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
        # List comprehensions make everything a tad faster than for loops
        K1 = self.__kernel_matrix(s, s)
        ss = [np.tile(s_, (self.s.shape[1], 1)).T for s_ in zip(*s)]
        K2 = [np.hstack(self.__kernel_matrix(ss_, self.s)) for ss_ in ss]
        xi = np.array([k @ self._mean_estimator for k in K2]).T
        sigma = np.array([self.alpha * (k1 - k2 @ self._estimator @ k2.T) for k1, k2 in zip(K1, K2)]).T
        self._logger.debug("KMP predict done.")
        if compute_KL:
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
        num_points = min(trajectory1_means.shape[1], trajectory2_means.shape[1])
        kl_divergences = np.zeros(num_points)

        for i in range(num_points):
            kl_divergences[i] = self.multivariate_kl_divergence(
                trajectory1_means[:,i], trajectory1_covariances[:,:,i],
                trajectory2_means[:,i], trajectory2_covariances[:,:,i]
            )

        mean_kl_divergence = np.mean(kl_divergences)
        return mean_kl_divergence