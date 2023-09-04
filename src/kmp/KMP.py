import copy
import logging
import math
import numpy as np

from numpy.linalg import inv, norm
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
        ktt = np.exp(-self.sigma_f*norm(t1-t2)**2)
        ktdt_tmp = np.exp(-self.sigma_f*norm(t1-t2dt)**2)
        ktdt = (ktdt_tmp - ktt)/dt
        kdtt_tmp = np.exp(-self.sigma_f*norm(t1dt-t2)**2)
        kdtt = (kdtt_tmp - ktt)/dt
        kdtdt_tmp = np.exp(-self.sigma_f*norm(t1dt-t2dt)**2)
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
        """Train the model by computing the estimator matrix inv(K+lambda*sigma).

        Parameters
        ----------
        X : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        mu : array-like of shape (n_output_features,n_samples)
            Array of output vectors
        var : array-like of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        self.s = copy.deepcopy(X)
        self.xi = copy.deepcopy(mu)
        self.sigma = copy.deepcopy(var)
        self.O = self.xi.shape[0]
        self.N = self.xi.shape[1]
        k = np.zeros((self.N*self.O, self.N*self.O))
        # Construct the estimator
        for i in range(self.N):
            for j in range(self.N):
                kernel = self.__kernel_matrix(self.s[:, i], self.s[:, j])
                k[i*self.O:(i+1)*self.O, j*self.O:(j+1)*self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    k[j*self.O:(j+1)*self.O, i*self.O:(i+1) * self.O] += self.l*self.sigma[:, :, i]
        self._estimator = inv(k)
        self._logger.info("KMP fit done.")
        
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
                k[:, i*self.O:(i+1)*self.O] = self.__kernel_matrix(s[:, j], self.s[:, i])
                for h in range(self.O):
                    Y[i*self.O+h] = self.xi[h, i]
            xi[:, j] = np.squeeze((k@self._estimator@Y.reshape(-1,1)))
            sigma[:, :, j] = self.alpha*(self.__kernel_matrix(s[:, j], s[:, j]) - k@self._estimator@k.T)
        self._logger.info("KMP predict done.")
        return xi, sigma
    