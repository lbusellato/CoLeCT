import csv
import numpy as np
import re

from os import listdir
from os.path import abspath, dirname, join
from src.datatypes import Point, Quaternion
from tslearn.metrics import soft_dtw_alignment


ROOT = dirname(dirname(dirname(abspath(__file__))))


def create_dataset(demonstrations_path: str = '', demonstration_regex: str = r'') -> None:
    """Process a set of demonstration recordings into an usable dataset

    Parameters
    ----------
    demonstrations_path : str, default = ''
        The path of the directory containing the demonstrations, relative to ROOT.
    demonstration_regex : str, default = r''
        Regex used to locate the relevant files in the path.
    """
    demonstrations_path = join(ROOT, demonstrations_path)
    files = [f for f in listdir(demonstrations_path) if re.match(demonstration_regex, f) is not None]
    files.sort()
    qa = []
    out = []
    for i, file in enumerate(files):
        t_cnt = 1
        t_dt = 0.001
        with open(join(demonstrations_path, file)) as csv_file:
            reader = csv.DictReader(csv_file)
            for j, row in enumerate(reader):
                if j % 2 == 0:
                    t = t_cnt*t_dt  # float(row['timestamp'])
                    t_cnt += 1
                    x = float(row['pos_x'])
                    y = float(row['pos_y'])
                    z = float(row['pos_z'])
                    w = float(row['quat_w'])
                    wx = float(row['quat_x'])
                    wy = float(row['quat_y'])
                    wz = float(row['quat_z'])
                    fx = float(row['force_x'])
                    fy = float(row['force_y'])
                    fz = float(row['force_z'])
                    mx = float(row['torque_x'])
                    my = float(row['torque_y'])
                    mz = float(row['torque_z'])
                    if not qa:
                        # First sample, recover the auxiliary quaternion
                        qa = Quaternion.from_array([w, wx, wy, wz])
                    quat = Quaternion.from_array([w, wx, wy, wz])
                    # Project to euclidean space
                    quat_eucl = (quat*~qa).log()
                    out.append(Point(t, x, y, z, quat, quat_eucl, fx, fy, fz, mx, my, mz))
            np.save(join(ROOT, demonstrations_path, f'dataset{i:02d}.npy'), out)
            out = []

def trim_datasets(datasets_path: str = '') -> None:
    """Remove any leading or trailing force-only samples in order to allow interpolating between the rest.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    """
    datasets_path = join(ROOT, datasets_path)
    regex = r'dataset(\d{2})\.npy'
    datasets = [f for f in listdir(datasets_path) if re.match(regex, f) is not None]
    datasets.sort()
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        # Figure out the indexes to slice the dataset with
        points = np.array([point.x for point in dataset])
        i = np.where(points != 0)[0][0]
        j = dataset.shape[0] - np.where(reversed(points) != 0)[0][0]
        trimmed_dataset = dataset[i:j]
        np.save(join(ROOT, datasets_path, file), trimmed_dataset)

def as_array(dataset):
    return np.vstack([point.as_array() for point in dataset])

def from_array(array):
   return [Point.from_array(row) for row in array.T]

def interpolate_datasets(datasets_path: str = ''):
    """Fill in the force-only samples with a linear interpolation between the previous and next full samples.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    """
    datasets_path = join(ROOT, datasets_path)
    regex = r'dataset(\d{2})\.npy'
    datasets = [f for f in listdir(datasets_path) if re.match(regex, f) is not None]
    datasets.sort()
    qa = None
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        interp_dataset = as_array(dataset)
        # Find the indices of the known points (non-zero rows)
        known_indices = np.nonzero(np.any(interp_dataset[:, 1:7] != 0, axis=1))[0]
        # Find the indices of the missing points (zero rows)
        missing_indices = np.nonzero(np.all(interp_dataset[:, 1:7] == 0, axis=1))[0]
        # Get the time, position, and orientation of the known points
        time_known = interp_dataset[known_indices, 0]
        position_known = interp_dataset[known_indices, 1:4]
        orientation_known = interp_dataset[known_indices, 4:8]
        # Interpolate the missing points
        time_missing = interp_dataset[missing_indices, 0]
        # Interpolate the position
        for i in range(3):
            interp_dataset[missing_indices, i + 1] = np.interp(time_missing, time_known, position_known[:, i])
        # Interpolate the orientation (quaternion)
        for i in range(4):
            interp_dataset[missing_indices, i + 4] = np.interp(time_missing, time_known, orientation_known[:, i])
        if qa is None:
            qa = Quaternion.from_array(orientation_known[0])
        for i in missing_indices:
            t, x, y, z, w, qx, qy, qz, qe1, qe2, qe3, fx, fy, fz, mx, my, mz = interp_dataset[i]
            quat_eucl = (Quaternion.from_array([w, qx, qy, qz])*~qa).log()
            dataset[i] = Point(t, x, y, z, 
                               Quaternion.from_array([w, qx, qy, qz]), quat_eucl, 
                               fx, fy, fz, mx, my, mz)
        np.save(join(ROOT, datasets_path, file), dataset)


def compute_alignment_path(cost_matrix):
    """Computes the warping path from a cost matrix.

    Parameters
    ----------
    cost_matrix : _type_
        The cost matrix computed with soft_dtw_alignment.
    """
    return [np.argmax(row) for row in cost_matrix.T]


def align_datasets(datasets_path: str = ''):
    """Align the demonstrations temporally using soft-DTW.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    """
    datasets_path = join(ROOT, datasets_path)
    regex = r'dataset(\d{2})\.npy'
    files = [f for f in listdir(datasets_path) if re.match(regex, f) is not None]
    files.sort()
    datasets = [np.load(join(datasets_path, file), allow_pickle=True) for file in files]
    np.save(join(ROOT, datasets_path, f'dataset00.npy'), datasets[0])
    reference = as_array(datasets[0])[:, 1:8]
    for i, dataset in enumerate(datasets):
        if i > 0:
            cost_matrix, _ = soft_dtw_alignment(as_array(dataset)[:, 1:8], reference, gamma=2.5)
            np.save(join(ROOT, datasets_path, f'dataset{i:02d}.npy'), dataset[compute_alignment_path(cost_matrix)])

def load_datasets(datasets_path: str='', regex = r'dataset(\d{2})\.npy') -> np.ndarray:
    """Load all datasets in the given folder.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    regex : str, default = r'dataset(\d{2})\.npy'
        Regex used to locate the relevant files in the path.

    Returns
    -------
    np.ndarray
        Array of Point arrays.
    """
    # Load the demonstrations
    datasets_path = join(ROOT, datasets_path)
    datasets = [f for f in listdir(datasets_path) if re.match(regex, f) is not None]
    datasets.sort()
    return [np.load(join(datasets_path, f), allow_pickle=True) for f in datasets]
