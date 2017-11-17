import logging

import math
import torch
import torch.nn as nn

from torch.autograd import Variable
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import pickle
# from pre_train.pickler import load_model, save_model
import numpy as np

from pre_train.mlp_torch import MLP#, transform, back_transform, load_model
# from .pre_train.mlp_torch import MLP, transform, back_transform
_logger = logging.getLogger(__name__)

class MyDriver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """

    def __init__(self, logdata=True):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        self.model = pickle.load(open('pre_train/models/mod_temporal_torch','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        self.logger = open('logger', 'w')
        t = pickle.load(open('pre_train/models/ustd_torch', 'rb'))
        # t = load_model(open('pre_train/models/ustd_torch', 'rb'))
        self.mu = t[0]
        self.std = t[1]
        # number of previous states used for the prediction
        self.history = 4
        self.past_sensors = []
        self.past_command = [np.zeros((49))]
        self.past_command[-1][0] = (1 - self.mu.data[2]) / self.std.data[2]
        self.past_command[-1][2] = (1 - self.mu.data[2]) / self.std.data[2]
        self.it = 0
    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None

    def carstate_matrix(self, carstate):
        m = np.zeros((25))
        m[0] = self.past_command[-1]['accel']
        m[1] = self.past_command[-1]['brake']
        m[2] = self.past_command[-1]['steer']
        m[3] = carstate.speed_x
        m[4] = carstate.distance_from_center
        m[5] = carstate.angle
        for i in range(6, 25):
            # print(i)
            m[i] = carstate.distances_from_edge[i-6]
        return m.reshape(1, -1)

    def carstate_matrix2(self, carstate):
        m = np.zeros((48))
        m[0] = self.past_command[-1][0]
        m[1] = self.past_command[-1][1]
        m[2] = self.past_command[-1][2]
        m[3] = self.past_command[-1][3]
        m[4] = self.past_command[-1][4]
        m[5] = 0
        m[6] = carstate.angle
        m[7] = carstate.current_lap_time
        m[8] = carstate.damage
        m[9] = carstate.distance_from_start
        m[10] = carstate.distance_raced
        m[11] = carstate.fuel
        m[12] = carstate.last_lap_time
        m[13] = carstate.race_position
        m[14] = carstate.rpm
        m[15] = carstate.speed_x
        m[16] = carstate.speed_y
        m[17] = carstate.speed_z
        for i in range(18, 36):
            m[i] = carstate.distances_from_edge[i-18]
        m[36] = carstate.distance_from_center
        for i in range(37, 41):
            m[i] = carstate.wheel_velocities[i-37]
        m[41] = carstate.z
        for i in range(42, 47):
            m[i] = carstate.focused_distances_from_edge[i-42]

        return m

    def transform(self, y):
        x = y.copy()
        x -= self.u
        # print("before std", x)
        for i in range(x.shape[1]):
            if self.std[i] != 0:
                x[:, i] /= self.std[i]
        # print('after std', x)
        return x
    def back_transform(self, y):
        x = y.copy()
        # print('before back std', x)
        for i in range(x.shape[1]):
            if self.std[i] != 0:
                x[:, i] *= self.std[i]
        x += self.u[:x.shape[1]]
        # print('after back std', x)
        return x
    def drive(self, carstate: State) -> Command:
        """
        # Produces driving command in response to newly received car state.
        #
        # This is a dummy driving routine, very dumb and not really considering a
        # lot of inputs. But it will get the car (if not disturbed by other
        # drivers) successfully driven along the race track.
        # """
        command = Command()
        # self.steer(carstate, 0.0, command)
        self.it += 1

        #multiply speeds by 3.6
        # wheel velocities
        features = self.carstate_matrix2(carstate).reshape(1,-1)
        # _logger.info(carstate)
        features = Variable(torch.FloatTensor(features))
        # features = Variable(torch.FloatTensor(features))
        t_features = self.model.transform(features, self.mu, self.std)
        outCommand = Command()
        if len(self.past_command) >= self.history:

            feat2 = t_features.data
            # print(type(feat2), type(self.past_command[-1]))
            for i in range(1, self.history):
                feat2 = torch.cat((torch.FloatTensor(self.past_command[-i]).view(1, -1), feat2), 1)
            feat2 = Variable(feat2)


            t_prediction = self.model(feat2)
            # t_prediction = self.model((torch.from_numpy(t_features)))

            prediction = self.model.back_transform(t_prediction, self.mu, self.std)

            prediction = prediction[0]

            prediction.data[0] = max(0, prediction.data[0])
            prediction.data[1] = max(0, prediction.data[1])
            prediction.data[3] = max(0, prediction.data[3])
            prediction.data[4] = max(0, prediction.data[4])
            prediction.data[0] = min(1, prediction.data[0])
            prediction.data[1] = min(1, prediction.data[1])
            prediction.data[3] = min(1, prediction.data[3])
            prediction.data[4] = min(1, prediction.data[4])
            prediction.data[2] = np.rint(prediction.data[2])
            prediction.data[2] = max(0, prediction.data[2])
            prediction.data[2] = min(1, prediction.data[2])
            outCommand.accelerator = prediction.data[0]

            if self.it > 500:
                outCommand.gear = prediction.data[2]
                outCommand.steering = prediction.data[3]
                outCommand.brake = prediction.data[1]
            else:
                outCommand.gear = 1
                outCommand.steering = 0
                outCommand.brake = 0
            outCommand.clutch = prediction.data[4]
            prediction = self.model.transform(prediction.view(1, -1), self.mu, self.std)[0]

            # _logger.info(outCommand)
            _logger.info(carstate)

            t_features[0, :5] = prediction[:5]
            self.past_command.append(t_features.data.numpy()[0, :])

        else:
            numpy_feat = t_features.data.numpy()
            numpy_feat[0, :5] = self.past_command[-1][:5]
            self.past_command.append(numpy_feat[0, :])


        # if self.data_logger:
        #         self.data_logger.log(carstate, command)

        # print(len(self.past_command))
        if len(self.past_command) > self.history:
            self.past_command = self.past_command[1:]
        return outCommand

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        # else:
        #     command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )
