import numpy as np
import quaternion


class Robot:
    """ Robot (Base class)

    Args:
        ip (str) : IP-address of robot
    """

    def __init__(self, ip):
        self._ip = ip

        # Common robot data
        self.velocity_ = np.array([0, 0, 0, 0, 0, 0])
        self.position_ = np.array([0, 0, 0])
        self.rotation_ = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        self.ft_ = np.array([0, 0, 0, 0, 0, 0])

    def receive_data(self):
        """
        receive_data callback function
        """
        raise NotImplementedError("receive_data() is not implemented in Robot base class")

    @property
    def velocity(self):
        """velocity property"""
        return self.velocity_

    @velocity.setter
    def velocity(self, value):
        self.velocity_ = value

    @property
    def position(self):
        """position property"""
        return self.position_

    @position.setter
    def position(self, value):
        self.position_ = value

    @property
    def rotation(self):
        """rotation property"""
        return self.rotation_

    @rotation.setter
    def rotation(self, value):
        self.rotation_ = value

    @property
    def ft(self):
        """ft property"""
        return self.ft_

    @ft.setter
    def ft(self, value):
        self.ft_ = value
