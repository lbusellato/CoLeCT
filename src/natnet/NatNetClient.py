import copy
import ipaddress
import logging
import netifaces
import struct
import socket
import sys
import time

from typing import Callable, List
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

# Structs to speed up parsing
Vector3 = struct.Struct('<fff')
Quaternion = struct.Struct('<ffff')


class RigidBody:
    def __init__(self,
                 id: int = 0,
                 pos: List[float] = [0.0, 0.0, 0.0],
                 rot: List[float] = [0.0, 0.0, 0.0, 0.0]):
        self._id = id
        self._pos = pos
        self._rot = rot
        self._rb_marker_list = None
        self._error = None
        self._tracking_valid = None

    def as_string(self) -> str:
        out_str = f'\tID: {self.id}\n'
        out_str += f'\tPosition: {self.pos[0]:5.2f},{self.pos[1]:5.2f},{self.pos[2]:5.2f}\n'
        out_str += f'\tOrientation: \
            {self.rot[0]:5.2f},{self.rot[1]:5.2f},{self.rot[2]:5.2f},{self.rot[3]:5.2f}\n'
        return out_str

    @property
    def id(self) -> int:
        """
        Get the value of id.

        Returns:
            int: The value of id.
        """
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """
        Set the value of id.

        Args:
            value (int): The new value for id.
        """
        self._id = value

    @property
    def pos(self) -> List[float]:
        """
        Get the value of pos.

        Returns:
            List[float]: The value of pos.
        """
        return self._pos

    @pos.setter
    def pos(self, value: List[float]) -> None:
        """
        Set the value of pos.

        Args:
            value (List[float]): The new value for pos.
        """
        self._pos = value

    @property
    def rot(self) -> List[float]:
        """
        Get the value of rot.

        Returns:
            List[float]: The value of rot.
        """
        return self._rot

    @rot.setter
    def rot(self, value: List[float]) -> None:
        """
        Set the value of rot.

        Args:
            value (List[float]): The new value for rot.
        """
        self._rot = value

    @property
    def tracking_valid(self) -> bool:
        """
        Get the value of tracking_valid.

        Returns:
            bool: The value of tracking_valid.
        """
        return self._tracking_valid

    @tracking_valid.setter
    def tracking_valid(self, value: bool) -> None:
        """
        Set the value of tracking_valid.

        Args:
            value (bool): The new value for tracking_valid.
        """
        self._tracking_valid = value

    @property
    def error(self) -> float:
        """
        Get the value of error.

        Returns:
            float: The value of error.
        """
        return self._error

    @error.setter
    def error(self, value: float) -> None:
        """
        Set the value of error.

        Args:
            value (float): The new value for error.
        """
        self._error = value

    @property
    def rb_marker_list(self) -> List[str]:
        """
        Get the value of rb_marker_list.

        Returns:
            List[str]: The value of rb_marker_list.
        """
        return self._rb_marker_list

    @rb_marker_list.setter
    def rb_marker_list(self, value: List[str]) -> None:
        """
        Set the value of rb_marker_list.

        Args:
            value (List[str]): The new value for rb_marker_list.
        """
        self._rb_marker_list = value


class RigidBodyData:
    def __init__(self):
        self._rigid_body_list = []

    def add_rigid_body(self, rigid_body):
        self._rigid_body_list.append(copy.deepcopy(rigid_body))

    def as_string(self):
        out_str = f'\tRigid Body Count: {self.rigid_body_count}'
        for rigid_body in self.rigid_body_list:
            out_str += rigid_body.as_string()
        return out_str

    @property
    def rigid_body_count(self) -> int:
        """
        Get the value of rigid_body_count.

        Returns:
            int: The value of rigid_body_count.
        """
        return len(self._rigid_body_list)

    @property
    def rigid_body_list(self) -> List[RigidBody]:
        """
        Get the value of rigid_body_list.

        Returns:
            List[RigidBody]: The value of rigid_body_list.
        """
        return self._rigid_body_list


class NatNetClient():
    """NatNet SDK 3.1 client library. Only works for Unicast and with Motive 2.2.
    """

    # Client/server message ids
    NAT_CONNECT = 0
    NAT_SERVERINFO = 1
    NAT_REQUEST = 2
    NAT_RESPONSE = 3
    NAT_REQUEST_MODELDEF = 4
    NAT_MODELDEF = 5
    NAT_REQUEST_FRAMEOFDATA = 6
    NAT_FRAMEOFDATA = 7
    NAT_MESSAGESTRING = 8
    NAT_DISCONNECT = 9
    NAT_KEEPALIVE = 10
    NAT_UNRECOGNIZED_REQUEST = 100
    NAT_UNDEFINED = 999999.9999

    def __init__(self,
                 server_address: str = None,
                 client_address: str = None,
                 rigid_body_listener: Callable = None,
                 verbose: bool = False) -> None:
        """_summary_

        Parameters
        ----------
        server_address : str, default = None
            IP address of the server. If None, autodiscovery will be attempted. 
        client_address : str, default = None
            IP address of the client. If None, autodiscovery will be attempted.
        rigid_body_listener : Callable, default = None
            Callback method for when rigid body data is sent from the server.
        verbose : bool, default = False
            If True, logging will be set to DEBUG, otherwise it will be set to INFO.
        """
        # Set up logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # Client/server settings
        self._command_port = 1510
        self._data_port = 1511
        if client_address is None:
            client_address = self.client_address_autodiscovery()
        if server_address is None:
            server_address = self.server_address_autodiscovery()
        self._client_address = client_address
        self._server_address = server_address
        self._multicast_address = '239.255.42.99'
        self._rigid_body_listener = rigid_body_listener
        self._command_thread = None
        self._data_thread = None
        self._command_socket = None
        self._data_socket = None

    def start(self) -> None:
        is_running = self.run()
        if not is_running:
            self._logger.error("Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit as e:
                self._logger.error(e)
            finally:
                self._logger.info("Quitting")
                raise ConnectionError
        # Wait a bit then check the connection status
        time.sleep(1)
        if self.connected() is False:
            self._logger.error("Could not connect properly. Check that Motive streaming is running" +
                         " and that the server/client IPs are set correctly.")
            try:
                sys.exit(2)
            except SystemExit as e:
                self._logger.error(e)
            finally:
                self._logger.info("Quitting")
                self.shutdown()
                raise ConnectionError

    def run(self) -> bool:
        """Create the sockets and threads, and start them.

        Returns
        -------
        bool
            True if the connection is up, false otherwise.  
        """
        self.data_socket = self._create_data_socket()
        if self.data_socket is None:
            self._logger.error('Could not open data socket.')
            return False
        self.command_socket = self._create_command_socket()
        if self.command_socket is None:
            self._logger.error('Could not open command socket.')
            return False
        self.stop_threads = False
        self.data_thread = Thread(
            target=self._data_thread_function, args=(lambda: self.stop_threads, ), daemon=True)
        self.command_thread = Thread(
            target=self._command_thread_function, args=(lambda: self.stop_threads, ), daemon=True)
        self.data_thread.start()
        self.command_thread.start()
        self._send_request(self.NAT_CONNECT, '')
        return True

    def shutdown(self) -> None:
        """Close the sockets and join the threads.
        """
        self._logger.info('Shutdown called')
        self.stop_threads = True
        self.command_socket.close()
        self.data_socket.close()
        try:
            self.command_thread.join(timeout=1.0)
            self.data_thread.join(timeout=1.0)
        except RuntimeError as e:
            self._logger.error(f'Joining threads returned error: {e}')

    def connected(self) -> bool:
        """Check if the connection is up.

        Returns
        -------
        bool
            True if the connection is up, false otherwise.
        """
        if self.command_socket is None or self.data_socket is None:
            return False
        return True
    
    def server_address_autodiscovery(self) -> str:
        """Attempt to discover the OptiTrack server's IP.

        Parameters
        ----------
        client_address : str, default = '127.0.0.1'
            The IP of the client.

        Returns
        -------
        str
            The IP of the server.
        """
        # Get the network subnet range
        self._logger.warn("Sever autodiscovery is broken, do not use!")
        interfaces = netifaces.interfaces()
        network = None
        for interface in interfaces:
            if interface != 'lo':
                if netifaces.AF_INET in netifaces.ifaddresses(interface):
                    addresses = netifaces.ifaddresses(interface)[netifaces.AF_INET]
                    if addresses:
                        ip = addresses[0]['addr']
                        subnet_mask = addresses[0]['netmask']
                        network = ipaddress.IPv4Network(f"{ip}/{subnet_mask}", strict=False)
        if network is None:
            self._logger.error(f'Command socket error: {e}')
            return None
        # Loop over the subnet range and try to contact the server
        for ip in ipaddress.IPv4Network(network):
            ip_address = str(ip)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(1)
            try:
                result = client_socket.connect_ex((ip_address, self.command_port))
                if result == 0:
                    self._logger.info(f"Discovered server IP: {ip_address}")
                    return ip_address
            except socket.error as e:
                self._logger.error(f'Command socket error: {e}')
            finally:
                client_socket.close()
        return None

    def client_address_autodiscovery(self) -> str:
        """Find out the IP address of the client machine.

        Returns
        -------
        str
            The client's IP address.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('10.254.254.254', 1))
            res = s.getsockname()[0]
        except Exception:
            res = '127.0.0.1'
        finally:
            s.close()
        return res

    def _send_request(self,
                      command_id: int,
                      command: str) -> int:
        """Send a command over the command socket.

        Parameters
        ----------
        in_socket : socket.socket
            The socket.
        command_id : int
            The command ID.
        command : str
            The command text.

        Returns
        -------
        int
            The number of sent bytes.

        Raises
        ------
        NotImplementedError
            Raised when a command other than NAT_CONNECT or NAT_REQUEST_FRAMEOFDATA is requested.
        """
        if command_id == self.NAT_CONNECT:
            command = 'Ping'
            packet_size = len(command) + 1
        elif command_id == self.NAT_REQUEST_FRAMEOFDATA or command_id == self.NAT_KEEPALIVE:
            command = ''
            packet_size = 0
        else:
            raise NotImplementedError(
                f'Command id {command_id} not implemented.')
        data = command_id.to_bytes(2, byteorder='little')
        data += packet_size.to_bytes(2, byteorder='little')
        data += command.encode('utf-8')
        data += b'\0'
        return self.command_socket.sendto(data, (self.server_address, self.command_port))

    def _create_command_socket(self) -> socket.socket:
        """Create the socket for command communication.

        Returns
        -------
        socket.socket
            The command socket.
        """
        command_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            command_socket.bind((self.client_address, 0))
        except socket.herror as e:
            self._logger.error(f'Command socket host error: {e}')
            return None
        except socket.gaierror as e:
            self._logger.error(f'Command socket address error: {e}')
            return None
        except socket.timeout as e:
            self._logger.error(f'{e}')
            self._logger.error(f'Command socket timeout.')
            return None
        except socket.error as e:
            self._logger.error(f'Command socket error: {e}')
            return None
        command_socket.settimeout(2.0)
        command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return command_socket

    def _create_data_socket(self) -> socket.socket:
        """Create the socket for data communication.

        Returns
        -------
        socket.socket
            The data socket.
        """
        data_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            data_socket.bind(('', 0))
        except socket.herror as e:
            self._logger.error(f'Data socket host error: {e}')
            return None
        except socket.gaierror as e:
            self._logger.error(f'Data socket address error: {e}')
            return None
        except socket.timeout as e:
            self._logger.error(f'Data socket timeout.')
            return None
        except socket.error as e:
            self._logger.error(f'Data socket error: {e}')
            return None
        value = socket.inet_aton(self.multicast_address) + \
            socket.inet_aton(self.client_address)
        data_socket.setsockopt(
            socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, value)
        return data_socket

    def _unpack_rigid_body(self, data: bytes, rb_num: int) -> tuple[int, RigidBody]:
        id = int.from_bytes(data[:4], byteorder='little')
        offset = 4
        self._logger.debug(f'Rigid Body {rb_num} ID: {id}')
        pos = Vector3.unpack(data[offset:offset+12])
        offset += 12
        self._logger.debug(f'\tPosition: {pos[0]}, {pos[1]}, {pos[2]}')
        rot = Quaternion.unpack(data[offset:offset+16])
        offset += 16
        self._logger.debug(
            f'\tOrientation: {rot[3]}, {rot[0]}, {rot[1]}, {rot[2]}')
        rigid_body = RigidBody(id=id, pos=pos, rot=rot)
        if self._rigid_body_listener is not None:
            self._rigid_body_listener(rigid_body)
        return offset, rigid_body

    def _unpack_rigid_body_data(self, data: bytes) -> RigidBodyData:
        rigid_body_data = RigidBodyData()
        data = memoryview(data)
        rigid_body_count = int.from_bytes(data[:4], byteorder='little')
        offset = 4
        self._logger.debug(f'Rigid Body Count: {rigid_body_count}')
        for i in range(rigid_body_count):
            offset_tmp, rigid_body = self._unpack_rigid_body(data[offset:], i)
            offset += offset_tmp
            rigid_body_data.add_rigid_body(rigid_body)
        return rigid_body_data

    def _process_message(self, data: bytes) -> None:
        """Process an incoming data packet.

        Parameters
        ----------
        data : bytes
            The data packet.
        """
        message_id = int.from_bytes(data[0:2], byteorder='little')
        offset = 4
        self._logger.debug('Begin Packet\n------------')
        if message_id == self.NAT_FRAMEOFDATA:
            self._logger.debug(f'Message ID: {message_id} (NAT_FRAMEOFDATA)')
            offset += 4  # Frame Prefix data
            marker_set_count = int.from_bytes(
                data[offset:offset+4], byteorder='little')
            offset += 4  # Marker Set count
            for _ in range(marker_set_count):
                model_name = bytes(data[offset:]).partition(b'\0')
                offset += len(model_name) + 1
                offset += 4  # Marker count
            unlabeled_markers_count = int.from_bytes(
                data[offset:offset+4], byteorder='little')
            offset += 4  # Unlabeled Markers count
            for _ in range(unlabeled_markers_count):
                offset += 12  # Unlabeled Marker position
            rigid_body_data = self._unpack_rigid_body_data(data[offset:])
            self._logger.debug(rigid_body_data.as_string())
        else:
            self._logger.debug(f'Ignored message {message_id}.')
        self._logger.debug('End Packet\n----------')

    def _command_thread_function(self, stop: Callable[[], bool]) -> bool:
        """The function associated to the command thread.

        Parameters
        ----------
        stop : Callable[[], bool]
            Thread stopping condition.

        Returns
        -------
        bool
            True if no error occurred, false otherwise.
        """
        self.command_socket.settimeout(2.0)
        data = bytearray(0)
        recv_buffer_size = 64*1024
        while not stop():
            try:
                data, _ = self.command_socket.recvfrom(recv_buffer_size)
            except socket.herror as e:
                self._logger.error(f'Command thread function: Command socket host error: {e}')
                return False
            except socket.gaierror as e:
                self._logger.error(f'Command thread function: Command socket address error: {e}')
                return False
            except socket.timeout as e:
                self._logger.error(f'Command thread function: Command socket timeout.')
                return False
            except socket.error as e:
                self._logger.error(f'Command thread function: Command socket error: {e}')
                return False
            if data:
                self._process_message(data)
            if not stop():
                self._send_request(self.NAT_KEEPALIVE, '')
            data = bytearray(0)
        return True

    def _data_thread_function(self, stop: Callable[[], bool]) -> bool:
        """The function associated to the data thread.

        Parameters
        ----------
        stop : Callable[[], bool]
            Thread stopping condition.

        Returns
        -------
        bool
            True if no error occurred, false otherwise.
        """
        data = bytearray(0)
        recv_buffer_size = 64*1024
        while not stop():
            try:
                data, _ = self.data_socket.recvfrom(recv_buffer_size)
            except socket.herror as e:
                self._logger.error(f'Data thread function: Data socket host error: {e}')
                return False
            except socket.gaierror as e:
                self._logger.error(f'Data thread function: Data socket address error: {e}')
                return False
            except socket.timeout as e:
                self._logger.error(f'Data thread function: Data socket timeout.')
                return False
            except socket.error as e:
                self._logger.error(f'Data thread function: Data socket error: {e}')
                return False
            if data:
                self._process_message(data)
            data = bytearray(0)
        return True

    @property
    def server_address(self) -> str:
        """
        Get the value of server_address.

        Returns:
            str: The value of server_address.
        """
        return self._server_address

    @server_address.setter
    def server_address(self, value: str) -> None:
        """
        Set the value of server_address.

        Args:
            value (str): The new value for server_address.
        """
        self._server_address = value

    @property
    def client_address(self) -> str:
        """
        Get the value of client_address.

        Returns:
            str: The value of client_address.
        """
        return self._client_address

    @client_address.setter
    def client_address(self, value: str) -> None:
        """
        Set the value of client_address.

        Args:
            value (str): The new value for client_address.
        """
        self._client_address = value

    @property
    def multicast_address(self) -> str:
        """
        Get the value of multicast_address.

        Returns:
            str: The value of multicast_address.
        """
        return self._multicast_address

    @multicast_address.setter
    def multicast_address(self, value: str) -> None:
        """
        Set the value of multicast_address.

        Args:
            value (str): The new value for multicast_address.
        """
        self._multicast_address = value

    @property
    def command_port(self) -> int:
        """
        Get the value of command_port.

        Returns:
            int: The value of command_port.
        """
        return self._command_port

    @command_port.setter
    def command_port(self, value: int) -> None:
        """
        Set the value of command_port.

        Args:
            value (int): The new value for command_port.
        """
        self._command_port = value

    @property
    def data_port(self) -> int:
        """
        Get the value of data_port.

        Returns:
            int: The value of data_port.
        """
        return self._data_port

    @data_port.setter
    def data_port(self, value: int) -> None:
        """
        Set the value of data_port.

        Args:
            value (int): The new value for data_port.
        """
        self._data_port = value

    @property
    def command_socket(self) -> socket.socket:
        """
        Get the value of command_socket.

        Returns:
            socket.socket: The value of command_socket.
        """
        return self._command_socket

    @command_socket.setter
    def command_socket(self, value: socket.socket) -> None:
        """
        Set the value of command_socket.

        Args:
            value (socket.socket): The new value for command_socket.
        """
        self._command_socket = value

    @property
    def data_socket(self) -> socket.socket:
        """
        Get the value of data_socket.

        Returns:
            socket.socket: The value of data_socket.
        """
        return self._data_socket

    @data_socket.setter
    def data_socket(self, value: socket.socket) -> None:
        """
        Set the value of data_socket.

        Args:
            value (socket.socket): The new value for data_socket.
        """
        self._data_socket = value

    @property
    def command_thread(self) -> Thread:
        """
        Get the value of command_thread.

        Returns:
            Thread: The value of command_thread.
        """
        return self._command_thread

    @command_thread.setter
    def command_thread(self, value: Thread) -> None:
        """
        Set the value of command_thread.

        Args:
            value (Thread): The new value for command_thread.
        """
        self._command_thread = value

    @property
    def data_thread(self) -> Thread:
        """
        Get the value of data_thread.

        Returns:
            Thread: The value of data_thread.
        """
        return self._data_thread

    @data_thread.setter
    def data_thread(self, value: Thread) -> None:
        """
        Set the value of data_thread.

        Args:
            value (Thread): The new value for data_thread.
        """
        self._data_thread = value
