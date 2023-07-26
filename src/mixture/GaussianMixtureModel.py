import logging
import numpy as np

from numpy.linalg import inv
from numpy.typing import ArrayLike
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

REALMIN = np.finfo(np.float64).tiny  # To avoid division by 0

class GaussianMixtureModel():
    def __init__(self,
                 n_components: int = 10,
                 n_demos: int = 5,
                 n_input_features: int = 1,
                 dt: float = 0.1) -> None:
        self.n_components = n_components
        self.n_demos = n_demos
        self.n_input_features = n_input_features
        self.dt = dt
        self.logger = logging.getLogger(__name__)
        self.initialized = False

    def init(self, data: ArrayLike) -> None:
        """Initialize the Gaussian Mixture Model by computing the priors, means and covariances of 
        each Gaussian component.
        
        Parameters
        ----------
        data : ArrayLike of shape (n_features, n_samples)
            The dataset to initialize the GMM with.
        """
        n_features, n_samples = data.shape
        diag_reg_factor = np.eye(n_features) * 1e-8
        # Clustering data to initialize the GMM
        labels = np.arange(n_samples//self.n_demos) % self.n_components
        labels.sort()
        labels = np.tile(labels, self.n_demos)
        # Initialize the arrays for the priors, means and covariances
        self.priors = np.zeros((self.n_components))
        self.means = np.zeros((n_features, self.n_components))
        self.covariances = np.zeros((n_features, n_features, self.n_components))
        # Initialize the GMM components
        for c in range(self.n_components):
            ids = [id for id, t in enumerate(labels) if t == c]
            self.priors[c] = len(ids)
            self.means[:, c] = np.mean(data[:, ids].T, axis=0)
            self.covariances[:, :, c] = np.cov(data[:, ids]) + diag_reg_factor
        # Normalize the priors
        self.priors = self.priors / np.sum(self.priors)
        self.initialized = True

    def gaussPDF(self, data: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> ArrayLike:
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
            data = data.reshape(-1,1)
            cov = np.array([[cov]])
        n_features, n_samples = data.shape
        data_cntr = data.T - np.tile(mean.T, (n_samples, 1))
        pdf = np.sum(data_cntr@inv(cov)*data_cntr, axis=1)
        pdf = np.exp(-0.5*pdf)/np.sqrt(np.abs(np.linalg.det(cov))*(2*np.pi)**n_features + REALMIN)
        return pdf

    def likelihood(self, data: ArrayLike) -> ArrayLike:
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
            likelihood[i, :] = self.priors[i]*self.gaussPDF(data, self.means[:, i], self.covariances[:,:,i])
        return likelihood

    def fit(self, data: ArrayLike) -> None:
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
        max_it = 100
        tol = 1e-4
        diag_reg_factor = np.eye(self.n_features)*1e-4
        LL = np.zeros((max_it))
        converged = False
        for it in range(max_it):
            # E step
            L = self.likelihood(data)
            gamma = L / np.tile(np.sum(L, axis=0) + REALMIN, (self.n_components, 1))
            gamma2 = gamma / np.tile(np.sum(gamma, axis=1, keepdims=True) + REALMIN, (1, n_samples))
            # M step
            for i in range(self.n_components):
                self.priors[i] = np.sum(gamma[i, :])/n_samples
                self.means[:, i] = np.dot(data, gamma2[i, :].T)
                data_cntr = data - np.tile(self.means[:, i].reshape(-1,1), (1, n_samples))
                self.covariances[:, :, i] = np.dot(np.dot(data_cntr, np.diag(gamma2[i, :])), data_cntr.T) + diag_reg_factor
            # Average log-likelihood
            LL[it] = np.sum(np.log(np.sum(L, axis=0))) / n_samples
            if it > min_it and (LL[it] - LL[it-1] < tol or it == max_it - 1):
                self.logger.info(f"EM converged in {it} steps.")
                converged = True
                break
        if not converged:
            self.logger.info("EM did not converge.")

    def predict(self, data: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
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
        I, N = data.shape
        n_output_features = self.n_features - I
        diag_reg_factor = np.eye(n_output_features)*1e-8
        # Initialize needed arrays
        mu_tmp = np.zeros((n_output_features, self.n_components))
        means = np.zeros((n_output_features, N))
        covariances = np.zeros((n_output_features, n_output_features, N))
        H = np.zeros((self.n_components, N))
        for t in range(N):
            # Activation weight
            for i in range(self.n_components):
                mu = self.means[:I,i]
                sigma = self.covariances[:I,:I,i]
                H[i,t] = self.priors[i] * self.gaussPDF(data[:,t], mu, sigma)
            H[:,t] /= np.sum(H[:,t] + REALMIN)
            # Conditional means
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:,:I,i]@inv(self.covariances[:I,:I,i])
                mu_tmp[:,i] = self.means[I:,i] + sigma_tmp@(data[:,t]-self.means[:I,i])
                means[:,t] += H[i,t]*mu_tmp[:,i]
            # Conditional covariances
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:,I:,i] - ((self.covariances[I:,:I,i]/self.covariances[:I,:I,i]).reshape(-1,1))@self.covariances[:I,I:,i].reshape(-1,1).T
                covariances[:,:,t] += H[i,t]*(sigma_tmp + np.outer(mu_tmp[:,i],mu_tmp[:,i]))
            covariances[:,:,t] += diag_reg_factor - np.outer(means[:,t],means[:,t])
        return means, covariances