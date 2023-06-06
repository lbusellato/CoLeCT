import numpy as np

from dataclasses import dataclass


@dataclass
class Point():
    """Dataclass that defines the representation of each point in the demonstration
    database.
    """
    time: float = 0.0
    pose: np.ndarray = np.zeros(6)
    twist: np.ndarray = np.zeros(6)
    quat: np.ndarray = np.zeros(4)
    quat_eucl: np.ndarray = np.zeros(3)
    wrench: np.ndarray = np.zeros(6)
