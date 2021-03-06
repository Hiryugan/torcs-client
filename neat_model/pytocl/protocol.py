import enum
import logging
import socket
import time
import pickle
import select
import sys
import os
import importlib
from pytocl.car import State as CarState
from pytocl.driver import Driver
from config_parser import Configurator, Parser
_logger = logging.getLogger(__name__)

# special messages from server:
MSG_IDENTIFIED = b'***identified***'
MSG_SHUTDOWN = b'***shutdown***'
MSG_RESTART = b'***restart***'

# timeout for socket connection in seconds and msec:
TO_SOCKET_SEC = 1
TO_SOCKET_MSEC = TO_SOCKET_SEC * 1000

def import_func(fitness_file_path):
    sys.path.append(os.path.dirname(fitness_file_path))
    p = os.path.basename(fitness_file_path).split('.')[0]
    imported = importlib.__import__(p, globals(), locals(), ['fitness', 'fitness_end'], 0)
    fitness_func = imported.fitness
    fitness_end_func = imported.fitness_end
    return fitness_func, fitness_end_func


class Client:
    """Client for TORCS racing car simulation with SCRC network server.

    Attributes:
        hostaddr (tuple): Tuple of hostname and port.
        port (int): Port number to connect, from 3001 to 3010 for ten clients.
        driver (Driver): Driving logic implementation.
        serializer (Serializer): Implementation of network data encoding.
        state (State): Runtime state of the client.
        socket (socket): UDP socket to server.
    """

    def __init__(self, parser, hostname='localhost', port=3001, *,
                 driver=None, serializer=None, fitness_file=None):

        self.parser = parser

        self.hostaddr = (hostname, port)
        self.driver = driver or Driver()
        self.serializer = serializer or Serializer()
        self.state = State.STOPPED
        self.socket = None
        self.datafile = os.path.join(parser.save_path, 'fitness_value.txt')
        _logger.debug('Initializing {}.'.format(self))
        self.crashed=False
        self.stuck=False
        self.fitness=0
        self.timespend=0
        self.stack=0
        self.count=0
        self.sumspeed=0
        self.speed=0
        self.position=0
        self.positionReward=0


        _logger.debug('Initializing {}.'.format(self))

    def __repr__(self):
        return '{s.__class__.__name__}({s.hostaddr!r}) -- {s.state.name}' \
            ''.format(s=self)
  
    def run(self):
        time.sleep(5)
        """Enters cyclic execution of the client network interface."""
        while True:
            if self.state is State.STOPPED:
                # if self.state is State.STOPPED or self.state is State.RESTARTED:
                _logger.debug('Starting cyclic execution.')

                self.state = State.STARTING

                try:
                    _logger.info('Registering driver client with server {}.'
                                 .format(self.hostaddr))
                    self._configure_udp_socket()
                    self._register_driver()
                    self.state = State.RUNNING
                    _logger.info('Connection successful.')

                except socket.error as ex:
                    _logger.error('Cannot connect to server: {}'.format(ex))
                    self.state = State.STOPPED

            while self.state is State.RUNNING:
                self._process_server_msg()

                # if self.state is not State.RESTARTED:
            if self.state is State.RESTARTED:
                break
                # TODO: save files

        _logger.info('Client stopped.')
        self.state = State.STOPPED
    def stop(self):
        """Exits cyclic client execution (asynchronously)."""
        if self.state is State.RUNNING:
            _logger.info('Disconnecting from racing server.')
            self.state = State.STOPPING
            self.driver.on_shutdown()

    def _configure_udp_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(TO_SOCKET_SEC)

    def _register_driver(self):
        """
        Sends driver's initialization data to server and waits for acceptance
        response.
        """

        #TODO LOAD NETWORK

        angles = self.driver.range_finder_angles
        assert len(angles) == 19, \
            'Inconsistent length {} of range of finder iterable.'.format(
                len(angles)
            )

        data = {'init': angles}
        buffer = self.serializer.encode(
            data,
            prefix='SCR-{}'.format(self.hostaddr[1])
        )

        _logger.info('Registering client.')

        connected = False
        while not connected and self.state is not State.STOPPING:
            try:
                _logger.debug('Sending init buffer {!r}.'.format(buffer))
                self.socket.sendto(buffer, self.hostaddr)

                buffer, _ = self.socket.recvfrom(TO_SOCKET_MSEC)
                _logger.debug('Received buffer {!r}.'.format(buffer))
                if MSG_IDENTIFIED in buffer:
                    _logger.debug('Server accepted connection.')
                    connected = True

                # _logger.debug('Sending init buffer {!r}.'.format(buffer))
                # self.socket.sendto(buffer, self.hostaddr)
                # self.socket.setblocking(0)
                # ready = select.select([self.socket], [], [], 5)
                # if ready[0]:
                #     buffer, _ = self.socket.recvfrom(TO_SOCKET_MSEC)
                # else:
                #     exit(42)
                # _logger.debug('Received buffer {!r}.'.format(buffer))
                # if MSG_IDENTIFIED in buffer:
                #     _logger.debug('Server accepted connection.')
                #     connected = True

            except socket.error as ex:
                _logger.debug('No connection to server yet ({}).'.format(ex))

    def _process_server_msg(self):

        try:

            buffer, _ = self.socket.recvfrom(TO_SOCKET_MSEC)
            # _logger.info(self.serializer.decode(buffer))
            # buffer = [MSG_RESTART]

            _logger.debug('Received buffer {!r}.'.format(buffer))
            MPS_PER_KMH = 1000 / 3600
            # print('we')
            if not buffer:
                return

            elif MSG_SHUTDOWN in buffer:
                _logger.info('Server requested shutdown.')
                with open(self.datafile,'w') as f:
                    fitt=str(self.fitness)
                    f.write(fitt)



            elif MSG_RESTART in buffer:
                _logger.info('Server requested restart of driver.')


                self.driver.on_restart()
                self.state = State.RESTARTED

                with open(self.datafile, 'w') as f:

                    fitt = str(self.fitness)
                    f.write(fitt)

                # self.state = State.STOPPED
                # time.sleep(5)

            else:
                sensor_dict = self.serializer.decode(buffer)
                carstate = CarState(sensor_dict)
                _logger.debug(carstate)

                command = self.driver.drive(carstate)

                self.crashed = False
                self.stuck = False
                if carstate.current_lap_time > 3:

                    if carstate.distance_from_center < -0.9 or carstate.distance_from_center > 0.9:
                        self.crashed = True
                # print(carstate.speed_x / MPS_PER_KMH, carstate.current_lap_time)
                if carstate.speed_x / MPS_PER_KMH < 2 and carstate.current_lap_time > 3:
                    self.stuck = True

                self.timespend = carstate.current_lap_time
                # import random
                fitness_function, fitness_end_func = import_func(fitness_file_path=self.parser.fitness_function_file)
                # self.fitness = random.random()
                if carstate.current_lap_time + carstate.last_lap_time > 130:
                    # print('time expired')
                    with open(self.datafile,'w') as f:

                        self.fitness = fitness_end_func(float(self.speed), float(carstate.distance_raced))
                        f.write(str(self.fitness))
                        # print('FITNEEEESSSSS ---> ', self.count, '  ', self.fitness)
                        command.meta = 1
                        # self.stop()
                if self.crashed:
                    # print('crashed')
                    with open(self.datafile,'w') as f:

                        self.fitness = self.fitness - 10000000
                        f.write(str(self.fitness))
                        # print('FITNEEEESSSSS ---> ', self.count, '  ', self.fitness)
                        command.meta = 1
                        # self.stop()
                elif self.stuck:
                    # print('stuck')
                    with open(self.datafile,'w') as f:
                        # print('stuck')
                        self.fitness = self.fitness - 1000000
                        # print('FITNEEEESSSSS ---> ', self.count, '  ', self.fitness)
                        f.write(str(self.fitness))
                        command.meta = 1
                        # self.stop()
                else:
                    self.count += 1

                    self.sumspeed += carstate.speed_x
                    self.speed = self.sumspeed / self.count
                    # print(carstate.race_position)
                    self.position = carstate.race_position

                    
                    self.fitness = fitness_function(float(self.speed), float(carstate.distance_raced))
                    # if carstate.last_lap_time == 0:
                    #     self.fitness = fitness_function(float(self.speed), float(carstate.distance_raced))
                    # else:
                    #     self.fitness = fitness_end_func(float(self.speed), float(carstate.distance_raced))


                    # print('--FITNEEEESSSSS ---> ', self.count, '  ', self.fitness)
                    # self.fitness += random.random()
                # print("this is the message",sensor_dict)

                _logger.debug(command)
                buffer = self.serializer.encode(command.actuator_dict)
                _logger.debug('Sending buffer {!r}.'.format(buffer))
                self.socket.sendto(buffer, self.hostaddr)

        except socket.error as ex:
            _logger.warning('Communication with server failed: {}.'.format(ex))

        except KeyboardInterrupt:
            _logger.info('User requested shutdown.')
            self.stop()


class State(enum.Enum):
    """The runtime state of the racing client."""

    # not connected to a racing server
    STOPPED = 1,
    # entering cyclic execution
    STARTING = 2,
    # connected to racing server and evaluating driver logic
    RUNNING = 3,
    # exiting cyclic execution loop
    STOPPING = 4,
    RESTARTED = 5


class Serializer:
    """Serializer for racing data dirctionary."""

    @staticmethod
    def encode(data, *, prefix=None):
        """Encodes data in given dictionary.

        Args:
            data (dict): Dictionary of payload to encode. Values are arrays of
                numbers.
            prefix (str|None): Optional prefix string.

        Returns:
            Bytes to be sent over the wire.
        """

        elements = []

        if prefix:
            elements.append(prefix)

        for k, v in data.items():
            if v and v[0] is not None:
                # string version of number array:
                vstr = map(lambda i: str(i), v)

                elements.append('({} {})'.format(k, ' '.join(vstr)))

        return ''.join(elements).encode()

    @staticmethod
    def decode(buff):
        """
        Decodes network representation of sensor data received from racing
        server.
        """
        d = {}
        s = buff.decode()

        pos = 0
        while len(s) > pos:
            start = s.find('(', pos)
            if start < 0:
                # end of list:
                break

            end = s.find(')', start + 1)
            if end < 0:
                _logger.warning('Opening brace at position {} not matched in '
                                'buffer {!r}.'.format(start, buff))
                break

            items = s[start + 1:end].split(' ')
            if len(items) < 2:
                _logger.warning(
                    'Buffer {!r} not holding proper key value pair.'.format(
                        buff
                    )
                )
            else:
                key = items[0]
                if len(items) == 2:
                    value = items[1]
                else:
                    value = items[1:]
                d[key] = value

            pos = end + 1

        return d
