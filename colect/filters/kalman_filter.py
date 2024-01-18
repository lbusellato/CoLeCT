from colect.filters.filter import Filter
from filterpy.kalman import KalmanFilter as KF
import numpy as np


class KalmanFilter(Filter):

    def __init__(self):
        super().__init__()

        # Initialize kalman filter for forces
        self.kalman_filter_ft = KF(dim_x=6, dim_z=6)
        self.kalman_filter_ft.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial state (forces)
        self.kalman_filter_ft.F = np.eye(6)  # state transition matrix
        self.kalman_filter_ft.H = np.eye(6)  # Measurement function
        self.kalman_filter_ft.P *= 1000.  # covariance matrix
        self.kalman_filter_ft.R = np.eye(6)
        self.kalman_filter_ft.R *= 10.0  # state uncertainty
        self.kalman_filter_ft.Q *= 0.001  # process uncertainty

    def process(self, ft):
        # Kalman Filter force-torque measurement
        self.kalman_filter_ft.predict()
        self.kalman_filter_ft.update(ft)
        filtered_ft = self.kalman_filter_ft.x
        self.f_filtered = np.array(filtered_ft[0:3])
        self.mu_filtered = np.array(filtered_ft[3:6])

    def reset(self):
        self.f_filtered = np.array([0, 0, 0])
        self.mu_filtered = np.array([0, 0, 0])
