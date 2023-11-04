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
    verbose : bool
        Enable/disable verbose output.
    """

    def __init__(
        self,
        l: float = 0.5,
        alpha: float = 40,
        sigma_f: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.l = l
        self.alpha = alpha
        self.sigma_f = sigma_f
        self.kl_divergence = None
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)

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
        """if len(t1) > 1:
            kernel_matrix = np.eye(self.O) * kernel(t1, t2)
        else:"""
        dt = 0.001
        ktt = kernel(t1, t2)
        ktdt_tmp = kernel(t1, t2 + dt)
        kdtt_tmp = kernel(t1 + dt, t2)
        kdtdt_tmp = kernel(t1 + dt, t2 + dt)
        # Components of the matrix
        ktdt = (ktdt_tmp - ktt) / dt
        kdtt = (kdtt_tmp - ktt) / dt
        kdtdt = (kdtdt_tmp - ktdt_tmp - kdtt_tmp + ktt) / dt**2
        # Fill the kernel matrix
        kernel_matrix = np.zeros((self.O, self.O))
        dim = self.O // 2
        for i in range(dim):
            kernel_matrix[i, i] = ktt
            kernel_matrix[i, i + dim] = ktdt
            kernel_matrix[i + dim, i] = kdtt
            kernel_matrix[i + dim, i + dim] = kdtdt
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
        self.alpha = 40#s.shape[1] / self.l
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
            xi[:, j] = np.squeeze((k @ self._estimator @ Y.reshape(-1, 1)))
            sigma[:, :, j] = self.alpha * (
                self.__kernel_matrix(s[:, j], s[:, j]) - k @ self._estimator @ k.T
            )
        self._logger.info("KMP predict done.")
        self.kl_divergence = self.KL_divergence(xi, sigma, self.xi, self.sigma)

        return xi, sigma

    def KL_divergence(self, xi, sigma, xi_ref, sigma_ref) -> float:
        kl_divs = []
        for i in range(self.N):
            # Create a multivariate distribution from data
            kmp_dist = multivariate_normal(xi[:, i], sigma[:, :, i])
            ref_dist = multivariate_normal(xi_ref[:, i], sigma_ref[:, :, i])
            # Evaluate the pdfs of the distributions
            kmp_pdf = kmp_dist.pdf(xi[:, i])
            ref_pdf = ref_dist.pdf(xi[:, i])
            # Compute the Kullback-Leibler Divergence
            #reg_factor = 1e-10 # Avoid getting huge numbers
            kl_div = ref_pdf * np.log(ref_pdf / kmp_pdf)
            kl_divs.append(kl_div)
        # Normalize, since kl divs can range wildly from tiny to huge numbers apprently
        kl_divs = np.array(kl_divs)
        kl_divs /= norm(kl_divs)
        # Compute an aggregate value
        return np.mean(kl_divs)