import numpy as np
from tslearn.metrics import soft_dtw_alignment

def compute_alignment_path(cost_matrix):
    alignment_path = [(0, 0)]  # Starting point
    i, j = 0, 0
    while len(alignment_path) < cost_matrix.shape[1]:
        if i < cost_matrix.shape[0] - 1 and j < cost_matrix.shape[1] - 1:
            if cost_matrix[i+1, j] >= cost_matrix[i, j+1] and cost_matrix[i+1, j] >= cost_matrix[i+1, j+1]:
                i += 1
            elif cost_matrix[i, j+1] >= cost_matrix[i+1, j] and cost_matrix[i, j+1] >= cost_matrix[i+1, j+1]:
                j += 1
            else:
                i += 1
                j += 1
        elif i < cost_matrix.shape[0] - 1:
            i += 1
        elif j < cost_matrix.shape[1] - 1:
            j += 1
        alignment_path.append((i, j))
    return np.array(alignment_path)

time_series1 = np.array([1, 2, 3, 4, 5])  # Replace with your actual time series
time_series2 = np.array([2, 4, 6, 8, 10, 12, 14])  # Replace with your actual time series

cost_matrix, distance = soft_dtw_alignment(time_series1, time_series2, gamma=0.1)
alignment_path = compute_alignment_path(cost_matrix)
print("Alignment path:", alignment_path)
warped_time_series = time_series1[alignment_path[:, 0]]

print("Soft-DTW distance:", distance)
print("Warped time series:", warped_time_series)
cost_matrix, distance = soft_dtw_alignment(warped_time_series, time_series2, gamma=0.1)
print(distance)