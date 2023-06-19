import numpy as np

from dataclasses import dataclass
from src.datatypes import Quaternion


@dataclass
class Point():
    """Dataclass that defines the representation of each point in the demonstration
    database.
    """
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rot: Quaternion = Quaternion.from_array([1.0, 0.0, 0.0, 0.0])
    rot_eucl: np.ndarray = np.array([0.0, 0.0, 0.0])
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """
        Get the value of position.
    
        Returns:
            np.ndarray: The value of position.
        """
        return np.array([self.x, self.y, self.z])
    
    @property
    def orientation(self) -> np.ndarray:
        """
        Get the value of orientation.
    
        Returns:
            np.ndarray: The value of orientation.
        """
        return self.rot.as_array()
    
    @property
    def force(self) -> np.ndarray:
        """
        Get the value of force.
    
        Returns:
            np.ndarray: The value of force.
        """
        return np.array([self.fx, self.fy, self.fz])
    
    @property
    def torque(self) -> np.ndarray:
        """
        Get the value of torque.
    
        Returns:
            np.ndarray: The value of torque.
        """
        return np.array([self.mx, self.my, self.mz])

    @property
    def wrench(self) -> np.ndarray:
        """
        Get the value of wrench.
    
        Returns:
            np.ndarray: The value of wrench.
        """
        return np.concatenate((self.force, self.torque))

    def as_array(self):
        return np.array([self.x, self.y, self.z, self.rot_eucl[0], self.rot_eucl[1], self.rot_eucl[2], self.fx, self.fy, self.fz, self.mx, self.my, self.mz])
    
    @classmethod
    def from_array(cls, array):
        return cls(*array)
