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
    def __init__(self, verbose: bool = False) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)
        self._frame_received = False
        self.nnc = NatNetClient(server_address='10.85.15.19',
                                client_address='10.94.26.237',
                                rigid_body_listener=self.nnc_callback)
        self.hex12 = HEX12(callback=self.hex12_callback)
        self.reading = np.empty(14)
        # Prepare the csv file
        recording_path = join(ROOT, 'recordings')
        recording_filename = time.strftime("%Y%m%d-%H%M%S") + '.csv'
        self.recording_file = join(recording_path, recording_filename)
        self.recording = False
        # TODO: This way of tracking time is a placeholder. Is real time needed?
        self.timestamp_counter = 1
        self.timestamp_dt = 0.001

    def run(self) -> int:
        help = 'Command list:\n\tr - Start/resume recording\n\tp - Pause recording\n\tq - Quit\n'
        self._logger.info(help)
        cmd = ''
        while cmd != 'r':
            cmd = input()
            if cmd == 'q':
                return 0
        header = ['timestamp', 'pos_x', 'pos_y', 'pos_z',
                  'quat_w', 'quat_x', 'quat_y', 'quat_z',
                  'force_x', 'force_y', 'force_z',
                  'torque_x', 'torque_y', 'torque_z']
        with open(self.recording_file, 'x') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        self.hex12.start()
        self.nnc.start()
        self.recording = True
        cmd = ''
        while cmd != 'q':
            cmd = input()
            if cmd == 'p':
                self.recording = False
            elif cmd == 'r':
                self.recording = True

    def hex12_callback(self, wrench: List[float]) -> None:
        """Handles recording force and torque data.

        Parameters
        ----------
        wrench: List[float]
            The list of force/torque values.
        """
        if self.frame_received and self.recording:
            self.reading[8:] = wrench
            with open(self.recording_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(self.reading)
            self.frame_received = False

    def nnc_callback(self, rigid_body: RigidBody) -> None:
        """Handles recording pose data.

        Parameters
        ----------
        rigid_body : RigidBody
            The Rigid Body received from Motive. 
        """
        if self.recording:
            self.reading[0] = self.timestamp_counter*self.timestamp_dt
            self.timestamp_counter += 1
            if not self.frame_received:
                self.reading[1:8] = [-1]*7
            else:
                self.reading[1:8] = rigid_body.pos + rigid_body.rot
                self.frame_received = True

    @property
    def frame_received(self) -> bool:
        """
        Get the value of frame_received.

        Returns:
            bool: The value of frame_received.
        """
        return self._frame_received

    @frame_received.setter
    def frame_received(self, value: bool) -> None:
        """
        Set the value of frame_received.

        Args:
            value (bool): The new value for frame_received.
        """
        self._frame_received = value


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    recorder = Recorder()
    recorder.run()


if __name__ == '__main__':
    main()
