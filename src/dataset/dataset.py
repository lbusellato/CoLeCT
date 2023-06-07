import csv
import numpy as np

from os import listdir
from os.path import abspath, dirname, isfile, join
from src.datatypes import Point, Quaternion

ROOT = dirname(dirname(dirname(abspath(__file__))))


def create_dataset(demonstrations_path: str = ROOT, subsample: int = 100):
    """Process a set of demonstration recordings into an usable dataset

    Parameters
    ----------
    demonstrations_path : str, default = ROOT
        _description_.
    subsample : int, default = 100
        _description_.

    Returns
    -------
    _type_
        _description_
    """
    files = [f for f in listdir(demonstrations_path) if '.npy' not in f and isfile(
        join(demonstrations_path, f))]
    qa = []
    out = []
    for file in files:
        with open(join(demonstrations_path, file)) as csv_file:
            reader = csv.DictReader(csv_file)
            for i, row in enumerate(reader):
                if i % subsample == 0:
                    t = float(row['timestamp'])
                    x = float(row['pos_x'])
                    y = float(row['pos_y'])
                    z = float(row['pos_z'])
                    w = float(row['quat_w'])
                    wx = float(row['quat_x'])
                    wy = float(row['quat_y'])
                    wz = float(row['quat_z'])
                    fx = float(row['force_x'])
                    fy = float(row['force_y'])
                    fz = float(row['force_z'])
                    mx = float(row['torque_x'])
                    my = float(row['torque_y'])
                    mz = float(row['torque_z'])
                    if i == 0:
                        # First sample, recover the auxiliary quaternion
                        qa = Quaternion.from_array([w, wx, wy, wz])
                    quat = Quaternion.from_array([w, wx, wy, wz])
                    # Project to euclidean space
                    quat_eucl = (quat*~qa).log()
                    out.append(
                        Point(t, x, y, z, quat, quat_eucl, fx, fy, fz, mx, my, mz))
    np.save(join(ROOT, demonstrations_path, "dataset.npy"), out)
    return out
