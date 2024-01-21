from colect.filters.filter import Filter
import math
import numpy as np


class LowPassFilter(Filter):

    def __init__(self, frequency=500.0):
        super().__init__()

        # Low pass Filter cut-off frequency
        self.frequency = frequency
        self.dt = 1 / self.frequency
        self.fc_f = 0.5
        self.LPF_alpha_f = self.calculate_lpf_alpha(self.fc_f, self.dt)

    def process(self, ft):
        self.f_filtered = self.lp_filter(ft[0:3], self.f_filtered, self.LPF_alpha_f)
        self.mu_filtered = self.lp_filter(ft[3:6], self.mu_filtered, self.LPF_alpha_f)

    def reset(self):
        self.f_filtered = np.array([0, 0, 0])
        self.mu_filtered = np.array([0, 0, 0])

    def set_lpf_cutoff_freq(self, cutoff_freq):
        """Set the Low-pass filter cutoff frequency of the LPF used for filtering the forces.

        Args:
            cutoff_freq (float): the cutoff frequency
        """
        self.LPF_alpha_f = self.calculate_lpf_alpha(cutoff_freq, self.dt)

    @staticmethod
    def lp_filter(filter_input, filter_state, lpf_alpha):
        """Low-pass filter

            Args:
              filter_input ([]): input to be filtered
              filter_state ([]): initial filter state
              lpf_alpha (float): LPF alpha

            Returns:
              filter_state : the filter state
            """
        filter_out = filter_state - (lpf_alpha * (filter_state - filter_input))
        return filter_out

    @staticmethod
    def calculate_lpf_alpha(cutoff_frequency, dt):
        """Low-pass filter

            Args:
              cutoff_frequency (float): cut
              dt (float): timestep dt

            Returns:
              LPF alpha (float)
            """
        return (2 * math.pi * dt * cutoff_frequency) / (2 * math.pi * dt * cutoff_frequency + 1)
