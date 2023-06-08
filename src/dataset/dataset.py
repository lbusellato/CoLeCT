import csv
import numpy as np

from os import listdir
from os.path import abspath, dirname, isfile, join
from src.datatypes import Point, Quaternion
from tslearn.metrics import soft_dtw_alignment


ROOT = dirname(dirname(dirname(abspath(__file__))))


def create_dataset(demonstrations_path: str = '') -> None:
    """Process a set of demonstration recordings into an usable dataset

    Parameters
    ----------
    demonstrations_path : str, default = ''
        The path of the directory containing the demonstrations, relative to ROOT.
    """
    files = [f for f in listdir(demonstrations_path) if '.npy' not in f and isfile(
        join(demonstrations_path, f))]
    qa = []
    out = []
    for file in files:
        t_cnt = 1
        t_dt = 0.001
        with open(join(demonstrations_path, file)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
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
                out.append(
                    Point(t, x, y, z, quat, quat_eucl, fx, fy, fz, mx, my, mz))
            np.save(join(ROOT, demonstrations_path, f'{file[:-4]}.npy'), out)
            out = []


def trim_datasets(datasets_path: str = '') -> None:
    """Remove any leading or trailing force-only samples in order to allow interpolating between the rest.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    """
    datasets_path = join(ROOT, datasets_path)
    datasets = [f for f in listdir(datasets_path) if '.csv' not in f and isfile(
        join(datasets_path, f))]
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        # Figure out the indexes to slice the dataset with
        for i, point in enumerate(dataset):
            if point.x != 0:
                break
        for j, point in enumerate(reversed(dataset)):
            if point.x != 0:
                j = dataset.shape[0] - j
                break
        trimmed_dataset = dataset[i:j]
        np.save(join(ROOT, datasets_path, file), trimmed_dataset)


def dataset_to_array(dataset):
    ret = []
    for point in dataset:
        ret.append([point.timestamp, point.x, point.y, point.z, point.rot.as_array()[0], point.rot.as_array()[
                   1], point.rot.as_array()[2], point.rot.as_array()[3], point.rot_eucl[0], point.rot_eucl[1], point.rot_eucl[2], point.fx, point.fy, point.fz, point.mx, point.my, point.mz])
    return np.array(ret)


def interpolate_datasets(datasets_path: str = ''):
    """Fill in the force-only samples with a linear interpolation between the previous and next full samples.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    """
    # TODO: this is not written that well...
    datasets_path = join(ROOT, datasets_path)
    datasets = [f for f in listdir(datasets_path) if '.csv' not in f and isfile(
        join(datasets_path, f))]
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        interp_dataset = dataset_to_array(dataset)
        # Find the indices of the known points (non-zero rows)
        known_indices = np.nonzero(
            np.any(interp_dataset[:, 1:7] != 0, axis=1))[0]
        # Find the indices of the missing points (zero rows)
        missing_indices = np.nonzero(
            np.all(interp_dataset[:, 1:7] == 0, axis=1))[0]
        # Get the time, position, and orientation of the known points
        time_known = interp_dataset[known_indices, 0]
        position_known = interp_dataset[known_indices, 1:4]
        orientation_known = interp_dataset[known_indices, 4:8]
        # Interpolate the missing points
        time_missing = interp_dataset[missing_indices, 0]
        # Interpolate the position
        for i in range(3):
            interp_dataset[missing_indices, i +
                           1] = np.interp(time_missing, time_known, position_known[:, i])
        # Interpolate the orientation (quaternion)
        for i in range(4):
            interp_dataset[missing_indices, i +
                           4] = np.interp(time_missing, time_known, orientation_known[:, i])
        for i in missing_indices:
            t, x, y, z, w, qx, qy, qz, qe1, qe2, qe3, fx, fy, fz, mx, my, mz = interp_dataset[
                i]
            new_point = Point(
                t, x, y, z, Quaternion.from_array([w, qx, qy, qz]), [qe1, qe2, qe3], fx, fy, fz, mx, my, mz)
            dataset[i] = new_point
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
    files = [f for f in listdir(datasets_path) if '.csv' not in f and 'dataset' not in f and isfile(
        join(datasets_path, f))]
    files.sort()
    datasets = [np.load(join(datasets_path, file), allow_pickle=True)
                for file in files]
    reference = [p.x for p in datasets[0]]
    for i, dataset in enumerate(datasets):
        if i > 0:
            s1 = [p.x for p in dataset]
            cost_matrix, _ = soft_dtw_alignment(s1, reference, gamma=0.1)
            np.save(join(ROOT, datasets_path,
                    f'new{i}.npy'), dataset[compute_alignment_path(cost_matrix)])
