import csv
import numpy as np

from os import listdir
from os.path import abspath, dirname, join
from colect.datatypes import Point, Quaternion
from tslearn.metrics import soft_dtw_alignment


ROOT = dirname(dirname(dirname(abspath(__file__))))

def create_dataset(demonstrations_path: str = '', subsample: int = 2) -> None:
    """Process a set of demonstration recordings into an usable dataset

    Parameters
    ----------
    demonstrations_path : str, default = ''
        The path of the directory containing the demonstrations, relative to ROOT.
    subsample : int, default = 2
        Subsampling to apply to the raw data. One every subsample-th data point will be preserved.
    """
    demonstrations_path = join(ROOT, demonstrations_path)
    files = [f for f in listdir(demonstrations_path) if f.endswith('.csv')]
    files.sort()
    qa = []
    out = []
    for i, file in enumerate(files):
        t_cnt = 1
        with open(join(demonstrations_path, file)) as csv_file:
            reader = csv.DictReader(csv_file)
            # Compute dt
            line_count = sum(1 for _ in reader)
            csv_file.seek(0)
            next(reader)
            first_row = next(reader)
            first_timestamp = float(first_row['timestamp'])
            last_row = None
            for last_row in reader:
                if float(last_row['timestamp']) != 0.0:
                    last_timestamp = float(last_row['timestamp'])
            t_dt = (last_timestamp - first_timestamp) / line_count
            csv_file.seek(0)
            next(reader)

            for j, row in enumerate(reader):
                time = t_cnt*t_dt
                t_cnt += 1
                if j % subsample == 0:
                    timestamp = float(row['timestamp'])
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
                    out.append(Point(timestamp, time, x, y, z, quat, quat_eucl, fx, fy, fz, mx, my, mz))
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
    datasets = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    datasets.sort()
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        # Figure out the indexes to slice the dataset with
        points = np.array([point.x for point in dataset])
        i = np.where(points != 0)[0][0]
        j = dataset.shape[0] - np.where(np.array(list(reversed(points))) != 0)[0][0]
        trimmed_dataset = dataset[i:j]
        np.save(join(ROOT, datasets_path, file), trimmed_dataset)

def as_array(dataset):
    return np.vstack([point.as_array() for point in dataset])

def from_array(array):
   return [Point.from_array(row) for row in array]

def interpolate_datasets(datasets_path: str = ''):
    """Fill in the force-only samples with a linear interpolation between the previous and next full samples.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    """
    datasets_path = join(ROOT, datasets_path)
    datasets = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    datasets.sort()
    qa = None
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        interp_dataset = as_array(dataset)
        # Find the indices of the known points (non-zero rows)
        known_indices = np.nonzero(np.any(interp_dataset[:, 2:8] != 0, axis=1))[0]
        # Find the indices of the missing points (zero rows)
        missing_indices = np.nonzero(np.all(interp_dataset[:, 2:8] == 0, axis=1))[0]
        # Get the time, position, and orientation of the known points
        time_known = interp_dataset[known_indices, 0]
        position_known = interp_dataset[known_indices, 2:5]
        orientation_known = interp_dataset[known_indices, 5:9]
        # Interpolate the missing points
        time_missing = interp_dataset[missing_indices, 0]
        # Interpolate the position
        for i in range(3):
            interp_dataset[missing_indices, i + 2] = np.interp(time_missing, time_known, position_known[:, i])
        # Interpolate the orientation (quaternion) using SLERP
        interp_dataset[missing_indices, 5:9] = slerp(time_missing, time_known, orientation_known)[:missing_indices.shape[0],:]
        if qa is None:
            qa = Quaternion.from_array(orientation_known[0])
        for i in missing_indices:
            timestamp, time, x, y, z, w, qx, qy, qz, _, _, _, fx, fy, fz, mx, my, mz = interp_dataset[i]
            quat_eucl = (Quaternion.from_array([w, qx, qy, qz])*~qa).log()
            dataset[i] = Point(timestamp, time, x, y, z, 
                               Quaternion.from_array([w, qx, qy, qz]), quat_eucl, 
                               fx, fy, fz, mx, my, mz)
        np.save(join(ROOT, datasets_path, file), dataset)
        
def slerp(time_missing : np.ndarray, time_known : np.ndarray, orientation_known : np.ndarray) -> np.ndarray:
    """Spherical linear interpolation of quaternions.

    Parameters
    ----------
    time_missing : np.ndarray
        Array of the timestamps of the missing orientations.
    time_known : np.ndarray
        Array of the timestamps of the known orientations.
    orientation_known : np.ndarray
        Array of known quaternions

    Returns
    -------
    np.ndarray
        The array of interpolated quaternions.
    """
    
    out = []
    for i in range(orientation_known.shape[0] - 1):
        # Initial and final quaternion, normalized
        q1 = orientation_known[i,:]
        q2 = orientation_known[i+1,:]
        q1 /= np.linalg.norm(q1)
        q2 /= np.linalg.norm(q2)
        # Compute how many interpolation steps we need
        n = np.where((time_missing >= time_known[i]) & (time_missing <= time_known[i + 1]))[0].shape[0]
        dot_product = np.dot(q1, q2)
        if dot_product < 0.0:
            # Switching the sign ensures that we take the shortest path on the quaternion sphere
            q1 = -q1
            dot_product = -dot_product
        if dot_product > 0.99:
            # They are basically the same quaternion
            for _ in range(n):
                out.append(q1)
        else:
            # Compute spherical linear interpolation between the quaternions
            theta = np.arccos(dot_product)
            # +2 to consider the initial and final quaternions, but only interpolate inbetween
            time = np.linspace(0, n + 2)
            for t in time[1:-1]:
                q_interpolated = (np.sin((1 - t) * theta) / np.sin(theta)) * q1 + (np.sin(t * theta) / np.sin(theta)) * q2
                out.append(q_interpolated / np.linalg.norm(q_interpolated))

    out = np.vstack(out)
    return out

def to_base_frame(datasets_path: str = '') -> None:
    """Transform the coordinates to the base frame of the robot
    """
    
    datasets_path = join(ROOT, datasets_path)
    datasets = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    datasets.sort()
    # Rotation of the robot base frame wrt Motive frame
    theta = -np.pi
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    # For Euclidean projection
    qa = None
    out = []
    UR5_base_position = np.array([1.095, 0.885, 1.5770])
    for file in datasets:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        new = as_array(dataset)
        # Translation
        new[:, 2:5] -= UR5_base_position
        # Rotation
        new[:, [3, 4]] = new[:, [4, 3]]
        new[:, 3] *= -1
        # Quaternions
        for i in range(new.shape[0]):
            old_quat = Quaternion.from_array(new[i, 5:9])
            old_quat_R = old_quat.as_rotation_matrix()
            new_quat_R = R@old_quat_R.T
            new_quat = Quaternion.from_rotation_matrix(new_quat_R)
            if qa is None:
                qa = new_quat
            new[i, 5:9] = new_quat.as_array()
            new[i, 9:12] = (new_quat*~qa).log()
        new = from_array(new)
        np.save(join(ROOT, datasets_path, file), new)
    return out

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
    files = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    files.sort()
    datasets = [np.load(join(datasets_path, file), allow_pickle=True) for file in files]
    np.save(join(ROOT, datasets_path, f'dataset00.npy'), datasets[0])
    reference = as_array(datasets[0])[:, 2:9]
    for i, dataset in enumerate(datasets):
        if i > 0:
            cost_matrix, _ = soft_dtw_alignment(as_array(dataset)[:, 2:9], reference, gamma=2.5)
            for j, point in enumerate(dataset):
                point.timestamp = (j + 1) * 0.1
            np.save(join(ROOT, datasets_path, f'dataset{i:02d}.npy'), dataset[compute_alignment_path(cost_matrix)])

def load_datasets(datasets_path: str='') -> np.ndarray:
    """Load all datasets in the given folder.

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.

    Returns
    -------
    np.ndarray
        Array of Point arrays.
    """
    # Load the demonstrations
    datasets_path = join(ROOT, datasets_path)
    datasets = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    datasets.sort()
    return [np.load(join(datasets_path, f), allow_pickle=True) for f in datasets]

def clip_datasets(datasets_path: str='', lower_cutoff: float=0.0, upper_cutoff: float=0.0):
    datasets_path = join(ROOT, datasets_path)
    files = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    files.sort()
    if lower_cutoff != upper_cutoff and lower_cutoff < upper_cutoff:
        for file in files:
            dataset = np.load(join(datasets_path, file), allow_pickle=True)
            new = np.vstack([p for p in dataset if p.time >= lower_cutoff and p.time <= upper_cutoff]).flatten()
            np.save(join(ROOT, datasets_path, file), new)

def check_quat_signs(datasets_path: str=''):
    datasets_path = join(ROOT, datasets_path)
    files = [f for f in listdir(datasets_path) if f.endswith('.npy')]
    files.sort()

    for i, file in enumerate(files):
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        if i == 0:
            reference = dataset[0]
            # Get the signs from the reference demonstration
            signs = [np.sign(reference.rot[0]), np.sign(reference.rot[1]), np.sign(reference.rot[2]), np.sign(reference.rot[3])]
        else:
            # Loop over the rest of the demonstrations and adjust the signs
            for p in dataset:
                rot = p.rot
                rot[0] *= signs[0] * np.sign(rot[0])
                rot[1] *= signs[1] * np.sign(rot[1])
                rot[2] *= signs[2] * np.sign(rot[2])
                rot[3] *= signs[3] * np.sign(rot[3])
                p.rot = rot
            np.save(join(ROOT, datasets_path, file), dataset)

