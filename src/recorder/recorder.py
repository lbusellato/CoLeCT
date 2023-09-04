import csv
import logging
import numpy as np
import time

from os.path import abspath, dirname, join
from src.hex12 import HEX12
from src.natnet import NatNetClient, RigidBody
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

ROOT = dirname(dirname(dirname(abspath(__file__))))


class Recorder():
    def __init__(self, server_address : str = '10.85.15.142', verbose : bool = False) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)
        self.nnc = NatNetClient(server_address='10.85.15.142',
                                rigid_body_listener=self.nnc_callback)
        self.hex12 = HEX12(callback=self.hex12_callback)
        self.reading = np.zeros(14)
        # Prepare the csv file
        recording_path = join(ROOT, 'recordings')
        recording_filename = time.strftime("%Y%m%d-%H%M%S") + '.csv'
        self.recording_file = join(recording_path, recording_filename)
        self.recording = False
        self.start_time = time.time()

    def run(self) -> int:
        help = 'Command list:\n\tr - Start recording\n\ts - Stop recording\n\tq - Quit\n'
        self._logger.info(help)
        cmd = ''
        while cmd != 'r':
            cmd = input()
            if cmd == 'q':
                return 0
        header = ['timestamp', 'pos_x', 'pos_y', 'pos_z',
                  'quat_x', 'quat_y', 'quat_z','quat_w', 
                  'force_x', 'force_y', 'force_z',
                  'torque_x', 'torque_y', 'torque_z']
        with open(self.recording_file, 'x') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        self.hex12.start()
        self.nnc.start()
        self.recording = True
        cmd = ''
        new_rec = False
        while cmd != 'q':
            cmd = input()
            if cmd == 'r':
                self.recording = True
                if new_rec:
                    recording_path = join(ROOT, 'recordings')
                    recording_filename = time.strftime("%Y%m%d-%H%M%S") + '.csv'
                    self.recording_file = join(recording_path, recording_filename)
                    self.start_time = time.time()
                    with open(self.recording_file, 'x') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
            elif cmd == 's':
                self.reading = np.zeros(14)
                self.recording = False
                new_rec = True

    def hex12_callback(self, wrench: List[float]) -> None:
        """Handles recording force and torque data.

        Parameters
        ----------
        wrench: List[float]
            The list of force/torque values.
        """
        if self.recording:
            self.reading[0] = round(time.time() - self.start_time, 6)
            self.reading[8:] = wrench
            with open(self.recording_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(self.reading)
            self.reading = np.zeros(14)

    def nnc_callback(self, rigid_body: RigidBody) -> None:
        """Handles recording pose data.

        Parameters
        ----------
        rigid_body : RigidBody
            The Rigid Body received from Motive. 
        """
        if self.recording:
            rot = np.array([rigid_body.rot[3],rigid_body.rot[0],rigid_body.rot[1],rigid_body.rot[2]])
            self.reading[1:8] = rigid_body.pos + rigid_body.rot
