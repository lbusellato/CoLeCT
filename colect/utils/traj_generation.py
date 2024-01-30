import matplotlib.pyplot as plt
import numpy as np

from colect.datatypes import Quaternion
from scipy.spatial.transform import Rotation, Slerp

def linear_traj_w_gauss_noise(start: np.ndarray, end: np.ndarray, n_points: int = 100, mean: float = 0.05, var: float = 0.05):
    """
    Generate a linear trajectory with Gaussian noise between two poses.

    Parameters:
    - start: Starting pose as a numpy array [position, quaternion].
    - end: Ending pose as a numpy array [position, quaternion].
    - n_points: Number of points in the trajectory, including start and end (default is 100).
    - mean: Mean of the Gaussian noise (default is 0.05).
    - var: Variance of the Gaussian noise (default is 0.05).

    Returns:
    - traj: List of poses along the trajectory, each pose represented as [position, quaternion].
    """

    # Extract position and quaternion from start and end poses
    start_pos, start_quat = start[:3], start[3:]
    end_pos, end_quat = end[:3], end[3:]

    # Generate linear trajectory between start and end
    positions = [start_pos + i/n_points * (end_pos - start_pos) for i in range(n_points)]
    start_rot = Rotation.from_quat(start_quat)
    end_rot = Rotation.from_quat(end_quat)
    key_rots = Rotation.concatenate([start_rot, end_rot])
    key_times = [0,1]
    slerp = Slerp(key_times, key_rots)
    rotations = slerp(np.arange(n_points)/n_points)

    # Add Gaussian noise to positions and orientations
    noise_positions = np.random.normal(loc=0, scale=var, size=(n_points, 3)) + mean
    noise_orientations = np.random.normal(loc=0, scale=var, size=(n_points, 4)) + mean

    # Apply noise to positions and orientations
    positions += noise_positions
    rotations = Rotation.as_quat(rotations)
    rotations += noise_orientations

    # Combine positions and orientations to form the trajectory
    traj = np.column_stack((positions, rotations))

    return traj

def linear_traj(start: np.ndarray, end: np.ndarray, n_points: int = 100, qa = None):
    """
    Generate a linear trajectory between two poses.

    Parameters:
    - start: Starting pose as a numpy array [position, quaternion].
    - end: Ending pose as a numpy array [position, quaternion].
    - n_points: Number of points in the trajectory, including start and end (default is 100).
    - qa: Auxiliary quaternion to use for projecting quaternions to Euclidean space

    Returns:
    - traj: List of poses along the trajectory, each pose represented as [position, quaternion].
    """

    # Extract position and quaternion from start and end poses
    if start.shape[0] > 3:
        start_pos, start_quat = start[:3], start[3:]
        end_pos, end_quat = end[:3], end[3:]
    else:
        start_pos = start[:3]
        end_pos = end[:3]

    # Generate linear trajectory between start and end
    positions = [start_pos + i/n_points * (end_pos - start_pos) for i in range(n_points)]
    if start.shape[0] > 3:
        start_rot = Rotation.from_quat(start_quat)
        end_rot = Rotation.from_quat(end_quat)
        key_rots = Rotation.concatenate([start_rot, end_rot])
        key_times = [0,1]
        slerp = Slerp(key_times, key_rots)
        rotations = slerp(np.arange(n_points)/n_points)
        rotations = Rotation.as_quat(rotations)

        if qa is not None:
            quats = []
            for quat in rotations:
                qx, qy, qz, w = quat
                quats.append((Quaternion.from_array([w, qx, qy, qz])*~qa).log())
            rotations = np.array(quats)


        # Combine positions and orientations to form the trajectory
        traj = np.column_stack((positions, rotations))
    else:
        traj = np.array(positions)

    return traj

def linear_traj_w_midpoint_stop(start: np.ndarray, end: np.ndarray, n_points: int = 250, n_stop: int=50, qa = None):
    """
    Generate a linear trajectory between two poses, with a stop at the midpoint

    Parameters:
    - start: Starting position as a numpy array.
    - end: Ending position as a numpy array. 
    - n_points: Number of points in the trajectory, including start and end.
    - n_stop: Number of points to stop the trajectory in.

    Returns:
    - traj: List of positions along the trajectory.
    """

    # Generate linear trajectory between start and end
    positions = np.array([start + i/n_points * (end - start) for i in range(n_points)])
    midpoint = positions[positions.shape[0]//2, :]
    positions_start = np.array([start + i/((n_points - n_stop) // 2) * (midpoint - start) for i in range((n_points - n_stop) // 2)])
    positions_middle = np.tile(midpoint, (n_stop, 1))
    positions_end = np.array([midpoint + i/((n_points - n_stop) // 2) * (end - midpoint) for i in range((n_points - n_stop) // 2)])
    traj = np.vstack((positions_start, positions_middle, positions_end))
    midpoint_idx = np.arange((n_points - n_stop) // 2, (n_points - n_stop) // 2 + n_stop)

    return traj, midpoint_idx

def force_step(fl: float=0.0, fh: float=1.0, nl: int=50, nh: int=50):
    low = np.ones((nl-nh)//2) * fl
    high = np.ones(nh) * fh
    return np.concatenate((low, high, low))

if __name__=="__main__":
    traj = force_step(-5, -20, 250, 85)
    plt.plot(traj)
    plt.show()
