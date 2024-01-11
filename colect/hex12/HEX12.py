import logging
import struct
import threading

from typing import Callable
from serial.serialutil import SerialException
from serial.tools import list_ports
from serial import Serial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)


class HEX12():
    """Handles collecting data from the HEX12 6 DOF force/torque sensor.
    """

    def __init__(self,
                 com_port: str = 'autodiscover',
                 baud_rate: int = 2000000,
                 callback: Callable = None,
                 serial_number: str = '377234603038',
                 verbose: bool = False) -> None:
        """Initialize communication with the sensor.

        Parameters
        ----------
        com_port : str, default = 'autodiscover'
            COM port the sensor is attached to. If set to `autodiscover`, discovery of the port will
            be attempted.
        baud_rate : int, default = 2000000
            The baud rate for the serial communication.
        serial_number : str, default = '377234603038'
            The serial number of the sensor. Used for the autodiscovery process.
        verbose : bool, default = False
            If True, logging will be set to DEBUG, otherwise it will be set to INFO.
        """
        # Set up logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # Set up the connection
        self._com_port = com_port
        self._baud_rate = baud_rate
        self._serial_number = serial_number
        self.connected = False
        # Set up the thread that handles data reading and unpacking
        self._stop = False
        self.data_thread = threading.Thread(
            target=self.data_thread_function, daemon=True)
        self.callback = callback

    def start(self) -> None:
        if self.com_port == 'autodiscover':
            self.com_port = self.autodiscover()
        try:
            self.ser = Serial(self.com_port, self.baud_rate, timeout=1)
        except SerialException as e:
            self._logger.error(f'Serial Exception occurred: {e}')
        self._logger.info(f'Connected to: {self.com_port}')
        self.connected = True
        self.data_thread.start()

    def stop(self) -> None:
        """Join the data thread.
        """
        try:
            self.data_thread.join(timeout=2.0)
        except RuntimeError as e:
            self._logger.error(f'Joining data thread failed: {e}')
        self.ser.close()

    def data_thread_function(self) -> None:
        """Read from the serial port and unpack the incoming sensor data.
        """
        while not self.stop:
            try:
                data = self.ser.read(4*7)
                try:
                    force_torque_values = struct.unpack('7f', data)
                    if not self.connected:
                        self._logger.info(f'Connected to {self.com_port}')
                        self.connected = True
                except struct.error:
                    self._logger.info(f'Lost connection to {self.com_port}')
                    self.connected = False
                if self.connected:
                    self.callback(force_torque_values[:-1])
                    out_str = '('
                    for ft in force_torque_values[:-1]:
                        out_str += f'{ft:5.2f} '
                    out_str += ')'
                    self._logger.debug(f'F/T sensor values: {out_str}')
            except Exception as e:
                self._logger.error(f'Exception occurred: {e}')
        self.stop()

    def autodiscover(self) -> str:
        """Attempt discovering the COM port the sensor is attached to. This works by checking each 
        COM port and seeing if the connected device's serial number matches the one of the sensor.

        Returns
        -------
        str
            The COM port the sensor is attached to, or an empty string if it wasn't found.

        Raises
        ------
        RuntimeError
            If no sensor is found.
        """
        ports = list_ports.comports()
        for port, _, hwid in ports:
            if 'SER=' + self.serial_number in hwid:
                self._logger.debug(f'Found sensor connected to {port}')
                return port
        raise RuntimeError(f'F/T sensor COM port autodiscovery failed.')

    @property
    def com_port(self) -> str:
        """
        Get the value of com_port.

        Returns:
            str: The value of com_port.
        """
        return self._com_port

    @com_port.setter
    def com_port(self, value: str) -> None:
        """
        Set the value of com_port.

        Args:
            value (str): The new value for com_port.
        """
        self._com_port = value

    @property
    def baud_rate(self) -> int:
        """
        Get the value of baud_rate.

        Returns:
            int: The value of baud_rate.
        """
        return self._baud_rate

    @baud_rate.setter
    def baud_rate(self, value: int) -> None:
        """
        Set the value of baud_rate.

        Args:
            value (int): The new value for baud_rate.
        """
        self._baud_rate = value

    @property
    def serial_number(self) -> str:
        """
        Get the value of serial_number.

        Returns:
            str: The value of serial_number.
        """
        return self._serial_number

    @serial_number.setter
    def serial_number(self, value: str) -> None:
        """
        Set the value of serial_number.

        Args:
            value (str): The new value for serial_number.
        """
        self._serial_number = value

    @property
    def connected(self) -> bool:
        """
        Get the value of connected.

        Returns:
            bool: The value of connected.
        """
        return self._connected

    @connected.setter
    def connected(self, value: bool) -> None:
        """
        Set the value of connected.

        Args:
            value (bool): The new value for connected.
        """
        self._connected = value

    @property
    def stop(self) -> bool:
        """
        Get the value of stop.

        Returns:
            bool: The value of stop.
        """
        return self._stop

    @stop.setter
    def stop(self, value: bool) -> None:
        """
        Set the value of stop.

        Args:
            value (bool): The new value for stop.
        """
        self._stop = value
