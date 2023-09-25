import logging
import numpy as np

from numpy.linalg import inv, det
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

REALMIN = np.finfo(np.float64).tiny  # To avoid division by 0


class GaussianMixtureModel():
    """Representation of a Gaussian Mixture Model probability distribution. The class allows for the estimation of the parameters od a Gaussian Mixture Model, specifically in the case of learning from demonstration.

    Parameters
    ----------
    n_components : int, default = 10
        Number of Gaussian components of the model.
    n_demos : int, default = 5
        Number of demonstrations in the training dataset.
    diag_reg_factor : float, default = 1e-5
        Non negative regularization factor added to the diagonal of the covariances to ensure they are positive.
    max_it : int, default = 100
        Maximum number of iterations of Expectation-Maximization to perform.
    tol : float, default = 1e-3
        Convergence threshold for Expectation-Maximization.
    """

    def __init__(self,
                 n_components: int = 10,
                 n_demos: int = 5,
                 diag_reg_factor: float = 1e-5,
                 max_it: int = 100,
                 tol: float = 1e-3) -> None:
        self.n_components = n_components
        self.n_demos = n_demos
        self.diag_reg_factor = diag_reg_factor
        self.max_it = max_it
        self.tol = tol
        self.initialized = False
        self.logger = logging.getLogger(__name__)

    def init(self, data: np.array) -> None:
        """Initialize the Gaussian Mixture Model by computing the priors, means and covariances of 
        each Gaussian component.

        Parameters
        ----------
        data : ArrayLike of shape (n_features, n_samples)
            The dataset to initialize the GMM with.
        """
        n_features, n_samples = data.shape
        diag_reg_factor = np.eye(n_features) * self.diag_reg_factor
        # Used to compute the BIC (source: sklearn)
        cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        mean_params = n_features * self.n_components
        self.n_params = int(cov_params + mean_params + self.n_components - 1)
        # Clustering data to initialize the GMM
        labels = np.arange(n_samples//self.n_demos) % self.n_components
        labels.sort()
        labels = np.tile(labels, self.n_demos)
        # Initialize the arrays for the priors, means and covariances
        self.priors = np.zeros((self.n_components))
        self.means = np.zeros((n_features, self.n_components))
        self.covariances = np.zeros(
            (n_features, n_features, self.n_components))
        # Initialize the GMM components
        for c in range(self.n_components):
            ids = [id for id, t in enumerate(labels) if t == c]
            self.priors[c] = len(ids)
            self.means[:, c] = np.mean(data[:, ids].T, axis=0)
            self.covariances[:, :, c] = np.cov(data[:, ids]) + diag_reg_factor
        # Normalize the priors
        self.priors = self.priors / np.sum(self.priors)
        self.initialized = True

    def gaussPDF(self, data: np.array, mean: np.array, cov: np.array) -> np.array:
        """Compute the Gaussian Probability Density Function for the Gaussian distribution with 
        the given mean and covariance, evaluated in the given data points.

        Parameters
        ----------
        data : ArrayLike of shape (n_features, n_samples)
            The dataset to evaluate the PDF in.
        mean : ArrayLike of shape (n_features)
            The mean of the Gaussian distribution.
        cov : ArrayLike of shape (n_features, n_features)
            The covariance of the Gaussian distribution.

        Returns
        -------
        pdf : ArrayLike of shape (n_samples)
            The likelihoods of the dataset.
        """
        if data.ndim == 1:
            # Univariate case, adjust the input shapes
            data = data.reshape(-1, 1)
        n_features, n_samples = data.shape
        data_cntr = data.T - np.tile(mean.T, (n_samples, 1))
        pdf = np.sum(data_cntr@inv(cov)*data_cntr, axis=1)
        pdf = np.exp(-0.5*pdf)/np.sqrt(np.abs(det(cov))
                                       * (2*np.pi)**n_features + REALMIN)
        return pdf

    def likelihood(self, data: np.array) -> np.array:
        """Compute the likelihood of the dataset.

        Parameters
        ----------
        data : ArrayLike of shape (n_features, n_samples)
            The dataset to evaluate the likelihood in.

        Returns
        -------
        likelihood : ArrayLike of shape (n_components, n_features)
            The computed likelihood.
        """
        likelihood = np.zeros((self.n_components, data.shape[1]))
        for i in range(self.n_components):
            likelihood[i, :] = self.priors[i] * \
                self.gaussPDF(
                    data, self.means[:, i], self.covariances[:, :, i])
        return likelihood

    def fit(self, data: np.array) -> Tuple[float, float]:
        """Use Expectation-Maximization (EM) to train the GMM.

        Parameters
        ----------
        data : ArrayLike of shape (n_features, n_samples)
            The dataset to train the model on.
        """
        if not self.initialized:
            self.init(data)
        self.n_features, n_samples = data.shape
        min_it = 5
        diag_reg_factor = np.eye(self.n_features)*self.diag_reg_factor
        LL = np.zeros((self.max_it))
        for it in range(self.max_it):
            # Expectation step
            L = self.likelihood(data)
            gamma = L / np.tile(np.sum(L, axis=0) + REALMIN,
                                (self.n_components, 1))
            gamma2 = gamma / \
                np.tile(np.sum(gamma, axis=1, keepdims=True) +
                        REALMIN, (1, n_samples))
            # Maximization step
            for i in range(self.n_components):
                self.priors[i] = np.sum(gamma[i, :])/n_samples
                self.means[:, i] = np.dot(data, gamma2[i, :].T)
                data_cntr = data - \
                    np.tile(self.means[:, i].reshape(-1, 1), (1, n_samples))
                self.covariances[:, :, i] = np.dot(
                    np.dot(data_cntr, np.diag(gamma2[i, :])), data_cntr.T) + diag_reg_factor
            # Average log-likelihood
            LL[it] = np.sum(np.log(np.sum(L, axis=0))) / n_samples
            if it > min_it and (LL[it] - LL[it-1] < self.tol or it == self.max_it - 1):
                self.score = self.bic(data, LL[it])
                self.logger.info(f"EM converged in {it} steps. BIC: {self.score}")
                return self.score, LL[it]
        # If we reached here, EM did not converge
        self.logger.info("EM did not converge.")
        return 0, 0

    def predict(self, data: np.array) -> Tuple[np.array, np.array]:
        """Use Gaussian Mixture Regression to predict mean and covariance of the given inputs

        Parameters
        ----------
        data : ArrayLike of shape (n_input_features, n_samples)

        Returns
        -------
        means : ArrayLike of shape (n_output_features, n_samples)
            The mean vectors associated to each input point.
        covariances : ArrayLike of shape (n_output_features, n_output_features, n_samples)
            The covariance matrices associated to each input point.
        """
        # Dimensionality of the inputs, number of points
        I, N = data.shape
        # Dimensionality of the outputs
        O = self.n_features - I
        diag_reg_factor = np.eye(O)*self.diag_reg_factor
        # Initialize needed arrays
        mu_tmp = np.zeros((O, self.n_components))
        means = np.zeros((O, N))
        covariances = np.zeros((O, O, N))
        H = np.zeros((self.n_components, N))
        for t in range(N):
            # Activation weight
            for i in range(self.n_components):
                mu = self.means[:I, i]
                sigma = self.covariances[:I, :I, i]
                H[i, t] = self.priors[i] * self.gaussPDF(data[:, t], mu, sigma)
            H[:, t] /= np.sum(H[:, t] + REALMIN)
            # Conditional means
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:, :I,
                                             i]@inv(self.covariances[:I, :I, i])
                mu_tmp[:, i] = self.means[I:, i] + \
                    sigma_tmp@(data[:, t]-self.means[:I, i])
                means[:, t] += H[i, t]*mu_tmp[:, i]
            # Conditional covariances
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:, I:, i] - \
                    self.covariances[I:, :I, i]@inv(
                        self.covariances[:I, :I, i])@self.covariances[:I, I:, i]
                covariances[:, :, t] += H[i, t] * \
                    (sigma_tmp + np.outer(mu_tmp[:, i], mu_tmp[:, i]))
            covariances[:, :, t] += diag_reg_factor - \
                np.outer(means[:, t], means[:, t])
        self.logger.info("GMR done.")
        return means, covariances

    def bic(self, X, avg_log_likelihood):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_features, n_samples)
            The input samples.
        avg_log_likelihood : float
            The average log likelihood of the model.

        Returns
        -------
        bic : float
            The lower the better.
        """
        _, n_samples = X.shape
        return -2 * avg_log_likelihood * n_samples + self.n_params * np.log(n_samples)
