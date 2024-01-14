import matplotlib.pyplot as plt
import numpy as np

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

def linear_traj(start: np.ndarray, end: np.ndarray, n_points: int = 100):
    """
    Generate a linear trajectory between two poses.

    Parameters:
    - start: Starting pose as a numpy array [position, quaternion].
    - end: Ending pose as a numpy array [position, quaternion].
    - n_points: Number of points in the trajectory, including start and end (default is 100).

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
    rotations = Rotation.as_quat(rotations)

    # Combine positions and orientations to form the trajectory
    traj = np.column_stack((positions, rotations))

    return traj

if __name__=='__main__':
    quat = np.array([0.985, -0.168, -0.009, -0.029])
    quat = quat / np.linalg.norm(quat)
    start_pose = np.array([-0.337, 0.285, -0.358,quat[0],quat[1],quat[2],quat[3]])
    end_pose = np.array([-0.437, 0.285, -0.358,quat[0],quat[1],quat[2],quat[3]])
    trajectory = linear_traj(start_pose, end_pose, n_points=100)
    fig, ax = plt.subplots(2,4,figsize=(10,6))
    ax[0,0].plot(trajectory[:,0])
    ax[0,1].plot(trajectory[:,1])
    ax[0,2].plot(trajectory[:,2])
    ax[0,3].axis('off')
    ax[1,0].plot(trajectory[:,3])
    ax[1,1].plot(trajectory[:,4])
    ax[1,2].plot(trajectory[:,5])
    ax[1,3].plot(trajectory[:,6])
    plt.show()