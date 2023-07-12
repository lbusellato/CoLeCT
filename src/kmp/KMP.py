import logging
import math
import numpy as np

from numpy.typing import ArrayLike
from typing import Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
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
    verbose : bool
        Enable/disable verbose output.
    """

    def __init__(self,
                 l: float = 0.5,
                 alpha: float = 40,
                 sigma_f: float = 1.0,
                 tol: float = 0.0005,
                 verbose: bool = False) -> None:
        self.trained = False
        self.l = l
        self.alpha = alpha
        self.sigma_f = sigma_f
        self.tol = tol
        self.kl_divergence = None
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)

    def set_waypoint(self,
                     s: ArrayLike,
                     xi: ArrayLike,
                     sigma: ArrayLike) -> None:
        """Adds a waypoint to the database, checking for conflicts.

        Parameters
        ----------
        s : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        xi : array-like of shape (n_output_features,n_samples)
            Array of output vectors
        sigma : array-like of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        for j in range(len(s)):
            # Loop over the reference database to find any conflicts
            min_dist = math.inf
            for i in range(self.N):
                dist = np.linalg.norm(self.s[:, i]-s[j])
                if dist < min_dist:
                    min_dist = dist
                    id = i
            if min_dist < self.tol:
                # Replace the conflicting point
                self.s[:, id] = s[j]
                self.xi[:, id] = xi[j]
                self.sigma[:, :, id] = sigma[j]
            else:
                # Add the new point to the database
                self.s = np.append(self.s, np.array(s[j]).reshape(1, -1))
                self.xi = np.append(self.xi, xi[j])
                self.sigma = np.append(self.sigma, sigma[j])
        # Refit the model with the new data
        self.fit(self.s, self.xi, self.sigma)

    def __kernel_matrix(self,
                        t1: float,
                        t2: float) -> ArrayLike:
        """Computes the kernel matrix for the given input pair.

        Parameters
        ----------
        t1 : float
            The first input.
        t2 : float
            The second input.

        Returns
        -------
        kernel : array-like of shape (n_features,n_features)
            The kernel matrix evaluated in the provided input pair.
        """
        dt = 0.001
        t1dt = t1 + dt
        t2dt = t2 + dt
        ktt = np.exp(-self.sigma_f*(t1-t2)**2)
        ktdt_tmp = np.exp(-self.sigma_f*(t1-t2dt)**2)
        ktdt = (ktdt_tmp - ktt)/dt
        kdtt_tmp = np.exp(-self.sigma_f*(t1dt-t2)**2)
        kdtt = (kdtt_tmp - ktt)/dt
        kdtdt_tmp = np.exp(-self.sigma_f*(t1dt-t2dt)**2)
        kdtdt = (kdtdt_tmp - ktdt_tmp - kdtt_tmp + ktt)/dt**2
        kernel_matrix = np.zeros((self.O,self.O))
        dim = self.O//2
        for i in range(dim):
            kernel_matrix[i,i] = ktt
            kernel_matrix[i, i+dim] = ktdt
            kernel_matrix[i+dim, i] = kdtt
            kernel_matrix[i+dim, i+dim] = kdtdt
        return kernel_matrix

    def fit(self,
            X: ArrayLike,
            mu: ArrayLike,
            var: ArrayLike) -> None:
        """"Train" the model by computing the estimator matrices for the mean (K+lambda*sigma)^-1 and 
        for the covariance (K+lambda_c*sigma)^-1. 

        Parameters
        ----------
        X : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        Y : array-like of shape (n_output_features,n_samples)
            Array of output vectors
        var : array-like of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        self.s = X.copy()
        self.xi = mu.copy()
        self.sigma = var.copy()
        self.O = self.xi.shape[0]
        self.N = self.xi.shape[1]
        k = np.zeros((self.N*self.O, self.N*self.O))
        # Construct the estimators
        for i in range(self.N):
            for j in range(self.N):
                kernel = self.__kernel_matrix(self.s[:, i], self.s[:, j])
                k[i*self.O:(i+1)*self.O, j*self.O:(j+1)*self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    a = self.l*self.sigma[:, :, i]
                    k[j*self.O:(j+1)*self.O, i*self.O:(i+1) * self.O] += self.l*self.sigma[:, :, i]
        self._estimator = np.linalg.inv(k)
        a = 0

    def predict(self, s: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Carry out a prediction on the mean and covariance associated to the given input.

        Parameters
        ----------
        s : array-like of shape (n_features,n_samples)
            The set of inputs to make a prediction of.

        Returns
        -------
        xi : array-like of shape (n_features,n_samples)
            The array of predicted means.

        sigma : array-like of shape (n_features,n_features,n_samples)
            The array of predicted covariance matrices.
        """
        xi = np.zeros((self.O, s.shape[1]))
        sigma = np.zeros((self.O, self.O, s.shape[1]))
        for j in range(s.shape[1]):
            k = np.zeros((self.O, self.N*self.O))
            Y = np.zeros(self.N*self.O)
            for i in range(self.N):
                k[:, i*self.O:(i+1) *
                  self.O] = self.__kernel_matrix(s[:, j], self.s[:, i])
                for h in range(self.O):
                    Y[i*self.O+h] = self.xi[h, i]
            xi[:, j] = k@self._estimator@Y
            sigma[:, :, j] = self.alpha*(self.__kernel_matrix(s[:, j], s[:, j]) - k@self._estimator@k.T)
        self.kl_divergence = self.KL_divergence(self.xi, xi)
        self._logger.info(f'KMP Done. KL : {self.kl_divergence}')
        return xi, sigma

    def KL_divergence(self, p: ArrayLike, q: ArrayLike) -> float:
        """Computes the Kullback-Leibler divergence for the model.

        Parameters
        ----------

        p : ArrayLike of shape (n_features, n_samples)
            The GMR trajectory.
        q : ArrayLike of shape (n_features, n_samples)
            The KMP trajectory.

        Returns
        -------
        float
            The computed KL divergence.
        """
        p_norm = p / np.sum(p, axis=0)
        q_norm = q / np.sum(q, axis=0)  
        # Some terms might produce nan values, fix that
        epsilon = 1e-10  
        p_norm[p_norm < epsilon] = epsilon
        q_norm[q_norm < epsilon] = epsilon  
        return np.sum(p_norm * np.log(np.divide(p_norm, q_norm)))
