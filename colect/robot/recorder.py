import logging
import colect.robot.rtde.rtde as rtde
import colect.robot.rtde.rtde_config as rtde_config
import colect.robot.rtde.csv_writer as csv_writer
import threading
import time
import numpy as np
import pandas as pd
from queue import Queue, Empty, Full
import os
from threading import Lock

_logger = logging.getLogger('colect')


class DataRecording:
    """ DataRecording class

    A data recording class, which can be used for recording data from an UR robot plus any arbitrary extra data you
    want to record from your application.

    Args:
        robot_ip (str): the IP address of the robot
        directory (str): optional directory to place the data recording in. (will be created if it doesn't exists)
        filename (str): name of the file to save the data to.
        frequency (float): frequency of the data recording. (defaults to 500Hz)
    """
    def __init__(self, robot_ip, directory, filename, frequency=500.0):
        self.robot_ip = robot_ip
        self.directory = directory
        self.filename = filename
        self.filepath = self.directory + '/' + self.filename
        self.frequency = frequency
        self.dt = 1.0 / frequency

        self.data_labels = []
        self.data = None
        self._data_lock = Lock()

        config_file = os.path.join(os.path.dirname(__file__), 'record_configuration.xml')
        conf = rtde_config.ConfigFile(config_file)
        self._output_names, self._output_types = conf.get_recipe('out')

        port = 30004
        self._con = rtde.RTDE(self.robot_ip, port)

        self._is_running = False
        self._keep_running = False
        self._write_thread = None

    def add_data(self, data):
        """ Add extra data to the data recording.

        Args:
            data (Union[list, numpy.ndarray, float]): the data to add to the recording as list, numpy array or float
        """
        if self._is_running:
            if len(self.data_labels) > 0 and self.data is not None:
                if len(self.data) >= len(self.data_labels):
                    with self._data_lock:
                        self.data.clear()
                if isinstance(data, list):
                    for elem in data:
                        with self._data_lock:
                            self.data.append(elem)
                elif isinstance(data, np.ndarray):
                    for elem in data.flatten().tolist():
                        with self._data_lock:
                            self.data.append(elem)
                else:
                    with self._data_lock:
                        self.data.append(data)
            else:
                _logger.warning('You must add data labels to the DataRecording before adding the actual data!')
                raise RuntimeError('You must add data labels to the DataRecording before adding the actual data!')
        else:
            _logger.warning('You tried to add data to a DataRecording that has not been started yet.')
            raise RuntimeError('You tried to add data to a DataRecording that has not been started yet.')

    def add_data_labels(self, data_labels):
        """ Add data labels for the extra data to the recording.

        Args:
            data_labels ([]): the data labels for the extra data of the recording.
        """
        self.data_labels = data_labels
        self.data = []

    def start(self):
        """ Start the data recording

        Will connect to the RTDE port of the robot and start a thread for receiving and writing data.
        """
        self._con.connect()

        # get controller version
        self._con.get_controller_version()

        # setup recipes
        if not self._con.send_output_setup(self._output_names, self._output_types, frequency=self.frequency):
            _logger.error('Unable to configure output')

        if self._con.is_connected():
            _logger.debug('DataRecorder RTDE is connected.')
        else:
            self._con.connect()

            # get controller version
            self._con.get_controller_version()

            # setup recipes
            if not self._con.send_output_setup(self._output_names, self._output_types, frequency=self.frequency):
                _logger.error('Unable to configure output')

            _logger.debug("DataRecorder RTDE re-connected.")

        # start RTDE data synchronization
        if not self._con.send_start():
            _logger.error('Unable to start synchronization')

        self._keep_running = True
        self._write_thread = threading.Thread(target=self._write, daemon=True)
        self._write_thread.start()
        _logger.debug("Recording started.")

    def stop(self):
        """ Stop the data recording

        Will terminate the thread that receives and writes data and finally disconnect the RTDE connection.
        """
        self._keep_running = False
        self._write_thread.join()
        self._con.disconnect()
        _logger.debug("Recorder stopped.")

    def _write(self):
        """
        Thread that receives data from the RTDE connection and writes data to the specified csv file.
        """
        self._is_running = True
        directory = os.path.dirname(self.filepath)
        if directory:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except FileExistsError:
                    # directory already exists
                    pass

        with open(self.filepath, 'w') as csvfile:
            writer = csv_writer.CSVWriter(csvfile, self._output_names, self._output_types)
            if self.data is not None:
                writer.writeheader_extra(self.data_labels)
            else:
                writer.writeheader()

            while self._keep_running:
                state = self._con.receive_buffered(False)
                if state is not None:
                    if self.data is not None:
                        if self.data:
                            with self._data_lock:
                                extra_data = self.data
                            writer.writerow_extra(state, extra_data)
                    else:
                        writer.writerow(state)

                # Small sleep to avoid thread starvation
                time.sleep(0.0001)  # 100 us

        self._is_running = False

    def verify_data_integrity(self, max_num_of_missed_samples=100, epsilon=1e-4):
        """ Verify the data integrity of the file associated with the data recording.

        The function takes a maximum number of missed samples that is allowed for the file data integrity to be
        considered valid. If there is more than the maximum number of missed samples in the recorded data
        the data integrity is considered invalid.

        Args:
            max_num_of_missed_samples (int): Maximum number of missed samples considered acceptable.
            epsilon (float): the tolerance to check the data timestamps with (defaults to 1e-4)
        """
        data = pd.read_csv(self.filepath, delimiter=" ")
        num_of_missed_samples = 0
        for i in range(1, len(data['timestamp'])):
            diff = abs(data['timestamp'][i] - data['timestamp'][i - 1])
            if diff > self.dt + epsilon:
                num_of_missed_samples += 1

        if num_of_missed_samples >= max_num_of_missed_samples:
            _logger.error('Data integrity: INVALID, Too many missing samples in the recorded data! missed: ' +
                          str(num_of_missed_samples) + ' out of ' + str(len(data['timestamp'])))
            return False
        else:
            _logger.info('Data integrity: VALID, missed: ' + str(num_of_missed_samples) +
                         ' out of ' + str(len(data['timestamp'])))
            return True

    def is_running(self):
        """
        Returns:
            bool: Whether the recorder is running or not. True for running, False for not running.
        """
        return self._is_running

    def get_filepath(self):
        """
        Returns:
            path to the file associated with the data recording + the filename.
        """
        return self.filepath

    def get_filename(self):
        """
        Returns:
            the filename of the file associated with the data recording.
        """
        return self.filename

    def get_directory(self):
        """
        Returns:
            the directory path to the file associated with the data recording.
        """
        return self.directory


class Recorder:
    def __init__(self, host, filename, frequency=500):
        self.__reading = False
        self.frequency = frequency
        self.filename = filename

        config_file = os.path.join(os.path.dirname(__file__), 'record_configuration.xml')
        conf = rtde_config.ConfigFile(config_file)
        self._output_names, self._output_types = conf.get_recipe('out')

        port = 30004
        self._con = rtde.RTDE(host, port)
        self._con.connect()

        # get controller version
        self._con.get_controller_version()

        # setup recipes
        if not self._con.send_output_setup(self._output_names, self._output_types, frequency=frequency):
            _logger.error('Unable to configure output')

        _logger.info("RTDE reader connected.")
        self._keep_running = False
        self._write_thread = None

    def set_filename(self, filename):
        self.filename = filename

    def get_filename(self):
        return self.filename

    def _write(self, filename, extra_names=None, extra_queue=None):
        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except FileExistsError:
                # directory already exists
                pass

        with open(filename, 'w') as csvfile:
            writer = csv_writer.CSVWriter(csvfile, self._output_names, self._output_types)
            extra_len = 0
            if extra_queue is not None:
                writer.writeheader_extra(extra_names)
                extra_len = len(extra_names)
            else:
                writer.writeheader()

            i = 1
            try:
                while self._keep_running:
                    state = self._con.receive(False)
                    if state is not None:
                        if extra_queue is not None:
                            if extra_queue.qsize() > 0:
                                extra_data = []
                                for i in range(0, extra_len):
                                    extra_data.append(extra_queue.get(timeout=0.1))
                                    extra_queue.task_done()
                                writer.writerow_extra(state, extra_data)
                        else:
                            writer.writerow(state)
                        i += 1
                    else:
                        _logger.warning("State is NONE.")

            # except Empty:
            #     logging.error('Queue empty')
            # except Full:
            #     logging.error('Queue full')
            except rtde.RTDEException:
                _logger.error('RTDE Exception.')
                self._con.disconnect()
                time.sleep(2)

    def start_writing(self, extra_names=None, extra_queue=None):
        _logger.info("Recorder started.")
        self._keep_running = True

        if self._con.is_connected():
            print('Recorder RTDE is connected.')
        else:
            self._con.connect()

            # get controller version
            self._con.get_controller_version()

            # setup recipes
            if not self._con.send_output_setup(self._output_names, self._output_types, frequency=self.frequency):
                _logger.error('Unable to configure output')

            _logger.info("RTDE reader re-connected.")

        # start RTDE data synchronization
        if not self._con.send_start():
            _logger.error('Unable to start synchronization')

        self._write_thread = threading.Thread(target=self._write, args=(self.filename, extra_names, extra_queue,),
                                              daemon=True)
        self._write_thread.start()

    def stop_writing(self):
        _logger.info("Recorder stopped.")
        self._keep_running = False
