import csv
import numpy as np
import re

from os import listdir
from os.path import abspath, dirname, join, isfile
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
        t_dt = 0.1
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
        # Interpolate the orientation (quaternion) using SLERP
        interp_dataset[missing_indices, 4:8] = slerp(time_missing, time_known, orientation_known)
        if qa is None:
            qa = Quaternion.from_array(orientation_known[0])
        for i in missing_indices:
            t, x, y, z, w, qx, qy, qz, _, _, _, fx, fy, fz, mx, my, mz = interp_dataset[i]
            quat_eucl = (Quaternion.from_array([w, qx, qy, qz])*~qa).log()
            dataset[i] = Point(t, x, y, z, 
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

def to_base_frame(datasets_path: str = '', base_frame_recording_path : str = '') -> None:
    """Transform the coordinates to the base frame of the robot

    Parameters
    ----------
    datasets_path : str, default = ''
        The path to the datasets, relative to ROOT.
    base_frame_recording_path : str, default = ''
        The path to the recording of the robot base frame, relative to ROOT.
    """
    # Recover the position of the base frame of the robot
    files = [f for f in listdir(base_frame_recording_path) if isfile(join(base_frame_recording_path,f))]
    base_file = files[0]
    UR5_base_position = np.zeros(3)
    with open(join(base_frame_recording_path, base_file)) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            UR5_base_position[0] = float(row['pos_x'])
            UR5_base_position[1] = float(row['pos_y'])
            UR5_base_position[2] = float(row['pos_z'])        
    # Recover the recordings
    datasets_path = join(ROOT, datasets_path)
    regex = r'dataset(\d{2})\.npy'
    files = [f for f in listdir(datasets_path) if re.match(regex, f) is not None]
    files.sort()
    # Rotation of the robot base frame wrt Motive frame
    theta = -np.pi
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    # For Euclidean projection
    qa = None
    for file in files:
        dataset = np.load(join(datasets_path, file), allow_pickle=True)
        new = as_array(dataset)
        # Translation
        new[:, 1:4] -= UR5_base_position
        # Rotation
        new[:, [2, 3]] = new[:, [3, 2]]
        new[:, 2] *= -1
        # Quaternions
        for i in range(new.shape[0]):
            old_quat = Quaternion.from_array(new[i, 4:8])
            old_quat_R = old_quat.as_rotation_matrix()
            new_quat_R = R@old_quat_R.T
            new_quat = Quaternion.from_rotation_matrix(new_quat_R)
            if qa is None:
                qa = new_quat
            new[i, 4:8] = new_quat.as_array()
            new[i, 8:11] = (new_quat*~qa).log()
        np.save(join(ROOT, datasets_path, file), from_array(new))

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
            for j, point in enumerate(dataset):
                point.timestamp = (j + 1) * 0.1
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
