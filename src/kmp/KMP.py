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
    lambda1 : int, default=0.1
        Lambda regularization factor for the mean minimization problem.
    lambda2 : float, default=1
        Lambda regularization factor for the covariance minimization problem.
    l : float, default=0.01
        Inner coefficient of the squared exponential term.Z
    sigma_f : float, default=1
        Outer coefficient of the squared exponential term.Z
    tol : float, default=0.0005
        Tolerance for the discrimination of conflicting points.
    priorities : array-like of shape (n_trajectories,), default=None
        Functions that map the input space into a priority value for trajectory superposition. The 
        sum of all priority functions evaluated in the same (any) input must be one.
    verbose : bool
        Enable/disable verbose output.
    """

    def __init__(self,
                 lambda1: float = 0.1,
                 lambda2: float = 1.0,
                 l: float = 0.01,
                 sigma_f: float = 1.0,
                 tol: float = 0.0005,
                 priorities: ArrayLike = None,
                 verbose: bool = False) -> None:
        if lambda1 <= 0:
            raise ValueError('lambda1 must be strictly positive.')
        if lambda2 <= 0:
            raise ValueError('lambda2 must be strictly positive.')
        if l <= 0:
            raise ValueError('l must be strictly positive.')
        if sigma_f <= 0:
            raise ValueError('sigma_f must be strictly positive.')
        if tol <= 0:
            raise ValueError('tol must be strictly positive.')
        self.trained = False
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l = l
        self.sigma_f = sigma_f
        self.tol = tol
        self.priorities = priorities
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
        return self.sigma_f*np.exp(-(1/self.l)*(t1-t2)**2)[0]*np.eye(self.O)

    def fit(self,
            X: ArrayLike,
            Y: ArrayLike,
            var: ArrayLike) -> None:
        """"Train" the model by computing the estimator matrices for the mean (K+lambda*sigma)^-1 and 
        for the covariance (K+lambda_c*sigma)^-1. The n_trajectories axis of the arguments is 
        considered only if the `self.priorities` parameter is not None.

        Parameters
        ----------
        X : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        Y : array-like of shape (n_trajectories,n_output_features,n_samples)
            Array of output vectors
        var : array-like of shape (n_trajectories,n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        if self.priorities is None:
            # Single trajectory
            self.s = X.copy()
            self.xi = Y.copy()
            self.sigma = var.copy()
            self.O = self.xi.shape[0]
            self.N = self.xi.shape[1]
        else:
            # Trajectory superposition
            L = len(Y)
            self.s = X.copy()
            self.xi = np.zeros_like(Y[0])
            self.sigma = np.zeros_like(var[0])
            self.O = self.xi.shape[0]
            self.N = self.xi.shape[1]
            # Compute covariances
            for n in range(self.N):
                for l in range(L):
                    self.sigma[:, :, n] += np.linalg.inv(
                        var[l][:, :, n]/self.priorities[l](self.s[:, n]))
                # Covariance = precision^-1
                self.sigma[:, :, n] = np.linalg.inv(self.sigma[:, :, n])
            # Compute means
            for n in range(self.N):
                for l in range(L):
                    self.xi[:, n] += np.linalg.inv(var[l][:, :, n] /
                                                   self.priorities[l](self.s[:, n]))@Y[l][:, n]
                self.xi[:, n] = self.sigma[:, :, n]@self.xi[:, n]
        k_mean = np.zeros((self.N*self.O, self.N*self.O))
        k_covariance = np.zeros((self.N*self.O, self.N*self.O))
        # Construct the estimators
        for i in range(self.N):
            for j in range(self.N):
                kernel = self.__kernel_matrix(self.s[:, i], self.s[:, j])
                k_mean[i*self.O:(i+1)*self.O, j*self.O:(j+1)*self.O] = kernel
                k_covariance[i*self.O:(i+1)*self.O, j *
                             self.O:(j+1)*self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    k_mean[j*self.O:(j+1)*self.O, i*self.O:(i+1)
                           * self.O] = kernel + self.lambda1*self.sigma[:, :, i]
                    k_covariance[j*self.O:(j+1)*self.O, i*self.O:(i+1)
                                 * self.O] = kernel + self.lambda2*self.sigma[:, :, i]
        self.__mean_estimator = np.linalg.inv(k_mean)
        self.__covariance_estimator = np.linalg.inv(k_covariance)

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
            xi[:, j] = k@self.__mean_estimator@Y
            sigma[:, :, j] = (self.N/self.lambda2)*(self.__kernel_matrix(s[:, j],
                                                                    s[:, j]) - k@self.__covariance_estimator@k.T)
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
