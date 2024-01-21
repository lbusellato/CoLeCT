import numpy as np
from enum import Enum, IntEnum


class FilterType(IntEnum):
    PASS_THROUGH = 0,
    LOW_PASS = 1,
    KALMAN = 2


class Filter:
    """Filter (Base class)

    The default filter is a pass-through filter, which means no filtering
    is done on the ft measurements.

    """

    def __init__(self):
        self.f_filtered = np.array([0, 0, 0])
        self.mu_filtered = np.array([0, 0, 0])

    def process(self, ft):
        # Default filter is a pass-through
        self.f_filtered = np.array(ft[0:3])
        self.mu_filtered = np.array(ft[3:6])

    def reset(self):
        self.f_filtered = np.array([0, 0, 0])
        self.mu_filtered = np.array([0, 0, 0])

    def get_filtered_force(self):
        return self.f_filtered

    def get_filtered_torque(self):
        return self.mu_filtered
