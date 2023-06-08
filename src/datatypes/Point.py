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
