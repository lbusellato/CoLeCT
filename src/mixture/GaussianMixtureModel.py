import logging
import numpy as np

from numpy.linalg import inv, det
from numpy.random import default_rng
from numpy.typing import ArrayLike
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

REALMIN = np.finfo(np.float64).tiny  # To avoid division by 0


class GaussianMixtureModel:
    """A Gaussian Mixture Model representation that allows for the extraction of the probabilistic 
    parameters of the underlying Gaussian mixture distribution to a set of demonstrations.

    based on: "On Learning, Representing and Generalizing a Task in a Humanoid Robot", 
    Calinon et al., 2007

    Parameters
    ----------
    n_components : int, default=8
        The number of mixture components. If `n_components_range` is not None, then all of its 
        elements are tried, and the one producing the smallest BIC score is kept.

    n_components_range : array_like, default=None
        Specifies the values to try when computing the optimal number of Gaussian components.

    n_demos : int, default=5
        The number of demonstrations in the database.

    max_it : int, default=100
        Maximum number of iterations for the EM algorithm.

    tol : float, default=1e-4
        Convergence criterion (lower bound for the average log-likelihood) for the EM algorithm.
        Must be strictly positive.

    model_selection_tol : float, default=25
        Tolerance for the selection of the optimal number of Gaussian components.    

    random_state : int, default=None
        Random seed for the random initialization of KMeans.

    reg_factor : float, default=1e-4
        Regularization factor added to the diagonal of the covariance matrices to ensure they are 
        positive definite. Must be strictly positive.

    Attributes
    ----------
    priors_ : array-like of shape (n_components,)
        The priors of each component of the mixture.

    means_ : array-like of shape (n_features, n_components)
        The mean of each mixture component.

    covariances_ : array-like of shape (n_features, n_features, n_components)
        The covariance of each mixture component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of steps of EM to reach convergence.

    n_input_features_ : int
        The input space dimension seen during fit().

    n_output_features_ : int
        The output space dimension seen during fit().

    n_features_ : int
        The sum of n_input_features_ and n_output_features_.

    n_components_ : int
        The number of Gaussian components used during fit().

    n_demos_ : int
        The number of demonstrations seen during fit().

    n_samples_ : int
        The number of samples of each demonstration seen during fit().
    """

    def __init__(self,
                 n_components: int = 8,
                 n_components_range: ArrayLike = None,
                 n_demos: int = 5,
                 max_it: int = 100,
                 tol: float = 1e-4,
                 model_selection_tol: float = 5,
                 random_state: int = None,
                 reg_factor: float = 1e-8) -> None:
        # Class attributes
        self.n_components_ = n_components
        if n_components_range is None:
            n_components_range = np.array([n_components])
        self.n_components_range = n_components_range
        self.n_demos_ = n_demos
        self.max_it = max_it
        self.tol = tol
        self.reg_factor = reg_factor
        self.model_selection_tol = model_selection_tol
        self.rng = default_rng(random_state)
        self._logger = logging.getLogger(__name__)

    def __pdf(self,
              X: ArrayLike,
              sigma: ArrayLike,
              mu: ArrayLike) -> ArrayLike:
        """Computes the Gaussian probability density function defined by the given mean and covariance,
        evaluated in the given points.

        Parameters
        ----------
        X : int or array-like of shape (n_features,n_samples)
            The data points in which the pdf is evaluated.
        sigma : int or array-like of shape (n_features,n_features)
            The variance or covariance matrix that defines the distribution.
        mu : int or array-like of shape (n_features)
            The mean vector that defines the distribution.

        Returns
        -------
        pdf : int or array-like of shape (n_features,n_samples)
            The computed probability density function values.
        """
        if X.ndim <= 1:
            # Univariate case
            X_cntr = X - mu
            pdf = X_cntr**2/sigma
            try:
                pdf = np.exp(-0.5*pdf)/np.sqrt(2*np.pi*np.abs(sigma))
            except ZeroDivisionError:
                pdf = np.exp(-0.5*pdf)/np.sqrt(2*np.pi*np.abs(sigma)+REALMIN)
        else:
            # Multivariate case
            X_cntr = X - np.tile(mu, (self.n_demos_*self.n_samples_, 1)).T
            pdf = np.sum(X_cntr.T@inv(sigma)*X_cntr.T, axis=1)
            try:
                pdf = np.exp(-0.5*pdf)/np.sqrt((2*np.pi) ** (self.n_features_)*np.abs(det(sigma)))
            except ZeroDivisionError:
                pdf = np.exp(-0.5*pdf)/np.sqrt((2*np.pi) ** (self.n_features_)*np.abs(det(sigma)) + REALMIN)
        return pdf

    def __likelihood(self,
                     X: ArrayLike,
                     sigma: ArrayLike,
                     mu: ArrayLike,
                     priors: ArrayLike,
                     n_components: int) -> ArrayLike:
        """Computes the likelihood of the Gaussian distribution defined by the given mean and covariance,
        evaluated in the given points.

        Parameters
        ----------
        X : int or array-like of shape (n_features,n_samples)
            The data points in which the likelihood is evaluated.
        sigma : int or array-like of shape (n_features,n_features,n_components)
            The variance or covariance matrix that defines the distribution.
        mu : int or array-like of shape (n_features,n_components)
            The mean vector that defines the distribution.
        priors : int or array-like of shape (n_components)
            The prior probabilities of each mixture component.
        n_components : int
            The number of mixture components.

        Returns
        -------
        array-like of shape (n_samples,n_components)
            The likelihood of each sample with respect to each Gaussian component.
        """
        return [priors[c]*self.__pdf(X, sigma[:, :, c], mu[:, c]) for c in range(n_components)]

    def __bic(self, X: ArrayLike, L: float, K: int) -> float:
        """Computes the Bayesian Information Criterion (BIC) score for the given data and number of
        Gaussian components.

        Parameters
        ----------
        X : array-like of shape (n_input_features, n_samples)
            The array of input vectors.
        L : float
            The average log-likelihood of the model.
        K : int
            The number of Gaussian components.

        Returns
        -------
        float
            The computed BIC score.
        """
        N = X.shape[1]
        D = X.shape[0]
        # Number of parameters
        n_p = (K-1)+K*(D+0.5*D*(D+1))
        return -2*L*N + n_p*np.log(N)

    def fit(self, X: ArrayLike, Y: ArrayLike) -> None:
        """Fit the model using the Expectation-Maximization algorithm.

        Parameters
        ----------
        X : array-like of shape (n_input_features, n_samples)
            The array of input vectors.
        Y : array-like of shape (n_output_features, n_samples)
            The array of output vectors.
        """
        # Set the relevant attributes
        self.n_input_features_ = X.shape[0]
        self.n_output_features_ = Y.shape[0]
        self.n_features_ = self.n_input_features_ + self.n_output_features_
        self.n_samples_ = int(Y.shape[1]/self.n_demos_)
        # Setup the dataset
        X = np.vstack([X, Y])
        # Figure out the optimal number of mixture components
        self.converged_ = False
        # Model selection
        bics = []
        ddbics = []
        if len(self.n_components_range) > 1:
            for n_components in self.n_components_range:
                # Setup the arrays for the priors, means and covariance matrices
                priors = np.zeros((n_components))
                means = np.zeros((self.n_features_, n_components))
                covariances = np.zeros(
                    (self.n_features_, self.n_features_, n_components))
                # Clusterization of the dataset
                n_samples = int(X.shape[1]/self.n_demos_)
                labels = np.tile((np.arange(n_samples)*n_components/n_samples).astype(int), self.n_demos_)
                # Initial guess for GMM
                for c in range(n_components):
                    ids = [i for i, val in enumerate(labels) if val == c]
                    # Compute priors, mean and the covariance matrix
                    priors[c] = len(ids)
                    means[:, c] = np.mean(X[:, ids].T, axis=0)
                    covariances[:, :, c] = np.cov(
                        X[:, ids]) + np.eye(self.n_features_)*self.reg_factor
                # Normalize the prior probabilities
                priors = priors/np.sum(priors)
                # Expectation-Maximization
                LL = np.zeros(self.max_it)
                for it in range(self.max_it):
                    # Expectation step
                    L = self.__likelihood(
                        X, covariances, means, priors, n_components)
                    # Pseudo posterior
                    try:
                        gamma = L/np.tile(np.sum(L, axis=0), (n_components, 1))
                    except ZeroDivisionError:
                        gamma = L/np.tile(np.sum(L, axis=0) + REALMIN, (n_components, 1))
                    gamma_mean = gamma / np.tile(np.sum(gamma, axis=1), (self.n_samples_*self.n_demos_, 1)).T
                    # Maximization step
                    for c in range(n_components):
                        # Update priors
                        priors[c] = np.sum(gamma[c]) / (self.n_samples_*self.n_demos_)
                        # Update mean
                        means[:, c] = X @ gamma_mean[c].T
                        # Update covariance
                        X_cntr = X.T - np.tile(means[:, c], (self.n_samples_*self.n_demos_, 1))
                        sigma = X_cntr.T @ np.diag(gamma_mean[c]) @ X_cntr
                        covariances[:, :, c] = sigma + np.eye(self.n_features_)*self.reg_factor
                    # Average log likelihood
                    LL[it] = np.mean(np.log(np.sum(L, axis=0)))
                    # Check convergence
                    if it > 0 and (it >= self.max_it or np.abs(LL[it]-LL[it-1]) < self.tol):
                        bics.append(self.__bic(X, LL[it], n_components))
                        if len(bics) > 1:
                            ddbics = np.gradient(np.gradient(bics))
                        break
        if len(ddbics) > 0:
            self.ddbics = np.abs(ddbics)
            n_components = np.argwhere(np.abs(ddbics) <= self.model_selection_tol)
            if n_components.shape[0] != 0:
                self.n_components_ = self.n_components_range[n_components[0]][0]
            else:
                self.n_components_ = self.n_components_range[np.argmin(np.abs(ddbics))]
        # Setup the arrays for the priors, means and covariance matrices
        priors = np.zeros((self.n_components_))
        means = np.zeros((self.n_features_, self.n_components_))
        covariances = np.zeros(
            (self.n_features_, self.n_features_, self.n_components_))
        # Clusterization of the dataset
        n_samples = int(X.shape[1]/self.n_demos_)
        labels = np.tile((np.arange(n_samples)*self.n_components_/n_samples).astype(int), self.n_demos_)
        # Initial guess for GMM
        for c in range(self.n_components_):
            ids = [i for i, val in enumerate(labels) if val == c]
            # Compute priors, mean and the covariance matrix
            priors[c] = len(ids)
            means[:, c] = np.mean(X[:, ids].T, axis=0)
            covariances[:, :, c] = np.cov(
                X[:, ids]) + np.eye(self.n_features_)*self.reg_factor
        # Normalize the prior probabilities
        priors = priors/np.sum(priors)
        # Expectation-Maximization
        LL = np.zeros(self.max_it)
        for it in range(self.max_it):
            # Expectation step
            L = self.__likelihood(X, covariances, means, priors, self.n_components_)
            # Pseudo posterior
            try:
                gamma = L / np.tile(np.sum(L, axis=0), (self.n_components_, 1))
            except ZeroDivisionError:
                gamma = L / np.tile(np.sum(L, axis=0)+REALMIN, (self.n_components_, 1))
            gamma_mean = gamma / np.tile(np.sum(gamma, axis=1), (self.n_samples_*self.n_demos_, 1)).T
            # Maximization step
            for c in range(self.n_components_):
                # Update priors
                priors[c] = np.sum(gamma[c])/(self.n_samples_*self.n_demos_)
                # Update mean
                means[:, c] = X @ gamma_mean[c].T
                # Update covariance
                X_cntr = X.T - np.tile(means[:, c], (self.n_samples_*self.n_demos_, 1))
                sigma = X_cntr.T @ np.diag(gamma_mean[c]) @ X_cntr
                covariances[:, :, c] = sigma + np.eye(self.n_features_)*self.reg_factor
            # Average log likelihood
            LL[it] = np.mean(np.log(np.sum(L, axis=0)))
            # Check convergence
            if it > 0 and (it >= self.max_it or np.abs(LL[it]-LL[it-1]) < self.tol):
                self._logger.info(
                    f'EM converged in {it} iterations. n_components: {self.n_components_}, BIC: {self.__bic(X,LL[it],self.n_components_)}')
                self.priors_ = priors
                self.covariances_ = covariances
                self.means_ = means
                self.converged_ = True
                self.n_iter = it
                break
        if not self.converged_:
            raise RuntimeError("EM algorithm did not converge")

    def predict(self, X: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Predict the expected output using Gaussian Mixture Regression.

        Parameters
        ----------
        X : array-like of shape (n_input_features, n_samples)
            The array of input vectors.

        Returns
        -------
        means : array-like of shape (n_input_features, n_samples)
            The array of mean vectors.

        covariances : array-like of shape (n_input_features, n_input_features, n_samples)
            The array of covariance matrices.
        """
        # Get the indexes of the input- and output-related quantities
        I = X.shape[0]
        N = X.shape[1]
        ts = list(range(I))  # Input indexes
        s = list(range(I, self.covariances_.shape[0]))  # Output indexes
        # Setup the output arrays
        means = np.zeros((len(s), N))
        covariances = np.zeros((len(s), len(s), N))
        # Perform regression
        mu_tmp = np.zeros((len(s), self.n_components_))
        beta = np.zeros((self.n_components_, N))
        for i in range(N):
            for t in ts:  # If the input is multidimensional, repeat the procedure for each dimension
                # Compute posteriors (Calinon eq. 11)
                for c in range(self.n_components_):
                    beta[c, i] = self.priors_[
                        c]*self.__pdf(X[t, i], self.covariances_[t, t, c], self.means_[t, c])
                # Normalize
                try:
                    beta[:, i] = beta[:, i]/np.sum(beta[:, i])
                except ZeroDivisionError:
                    beta[:, i] = beta[:, i]/np.sum(beta[:, i]+REALMIN)
                # Compute conditional means (Calinon eq. 10)
                for c in range(self.n_components_):
                    mu_tmp[:, c] = self.means_[s, c] + self.covariances_[s, t,
                                                                         c]*(X[t, i]-self.means_[t, c])/self.covariances_[t, t, c]
                    means[:, i] = means[:, i] + beta[c, i]*mu_tmp[:, c]
                # Compute conditional covariances (Calinon eq. 10)
                for c in range(self.n_components_):
                    gmr_sigma = self.covariances_[np.ix_(s, s, [c])][:, :, 0] - (np.reshape(self.covariances_[
                        s, t, c], (len(s), 1))/self.covariances_[t, t, c])*self.covariances_[t, s, c]
                    covariances[:, :, i] = covariances[:, :, i] + \
                        (beta[c, i]**2)*gmr_sigma
                reg_factor = np.eye(len(s))*self.reg_factor
                covariances[:, :, i] = covariances[:, :, i] + reg_factor
        self._logger.info('GMR done')
        return means, covariances
