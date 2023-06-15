import logging
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

from os import listdir
from os.path import dirname, abspath, join
from src.dataset import as_array
from src.mixture import GaussianMixtureModel
from time import perf_counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

ROOT = dirname(dirname(abspath(__file__)))


class single_point_task():
    def __init__(self, verbose: bool = True) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)
        # Load the demonstrations
        regex = r'dataset(\d{2})\.npy'
        datasets_path = 'demonstrations/single_point_task'
        datasets_path = join(ROOT, datasets_path)
        datasets = [f for f in listdir(
            datasets_path) if re.match(regex, f) is not None]
        datasets.sort()
        datasets = [np.load(join(datasets_path, f), allow_pickle=True)
                    for f in datasets]
        # Get the GMR reference trajectory
        subsample = 20
        H = len(datasets)  # Number of demonstrations
        N = len(datasets[0]) // subsample  # Length of each demonstration
        # Prepare the data for GMM/GMR
        dt = 0.001
        X = dt*np.tile(np.arange(N + 1), H).reshape(1, -1)
        Y = None
        for dataset in datasets:
            if Y is None:
                Y = as_array(dataset)[::subsample, 1:4]
            else:
                Y = np.vstack((Y, as_array(dataset)[::subsample, 1:4]))
        Y = Y.T
        start = perf_counter()
        gmm = GaussianMixtureModel(n_demos=H)
        gmm.fit(X, Y)
        elapsed = perf_counter() - start
        print(f'Fit took: {elapsed}')
        x = dt*np.arange(1, N + 1).reshape(1, -1)
        start = perf_counter()
        mu_pos, sigma_pos = gmm.predict(x)
        elapsed = perf_counter() - start
        print(f'Predict took: {elapsed}')
        fig, ax = plt.subplots(4, 3)
        for j in range(3):
            ax[0, j].plot(mu_pos[j, :])
            ax[0, j].grid()
        plt.show()
        input()


def main(task):
    if task == '1':
        single_point_task()
    input()


if __name__ == '__main__':
    task = sys.argv[-1] if len(sys.argv) > 1 else None
    main(task)
