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

    def position(self):
        return [self.x, self.y, self.z]

    def force(self):
        return [self.fx, self.fy, self.fz]

    def torque(self):
        return [self.mx, self.my, self.mz]

    def as_array(self):
        return [self.x, self.y, self.z, self.rot_eucl[0], self.rot_eucl[1], self.rot_eucl[2], self.fx, self.fy, self.fz, self.mx, self.my, self.mz]
