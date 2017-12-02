import logging

import math
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import pickle
# from pre_train.pickler import load_model, save_model
import numpy as np

from pre_train.mlp_torch_2 import MLP#, transform, back_transform, load_model
# from .pre_train.mlp_torch import MLP, transform, back_transform
_logger = logging.getLogger(__name__)

# class MyDriver(Driver):
class MyDriver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """

    def __init__(self, logdata=True):

        start = time.time()
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

        lst = pickle.load(open('pre_train/models/dimensions', 'rb'))
        self.output_dimensions = [3]#lst[0]
        self.state_dimensions = lst[1]
        self.input_dimensions = lst[2]

        self.models = dict()
        # for i in [0,3]:
        #     self.models[i] = pickle.load(open('pre_train/models/mod_temporal_torch' + '_' + str(i),'rb'))
        self.model3 = pickle.load(open('pre_train/models/mod_temporal_torch_3','rb'))
        # self.model1 = pickle.load(open('pre_train/models/mod_temporal_torch_1','rb'))
        # self.model0 = pickle.load(open('pre_train/models/mod_temporal_torch_0','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        self.logger = open('logger', 'w')
        t = pickle.load(open('pre_train/models/ustd_torch', 'rb'))
        # t = load_model(open('pre_train/models/ustd_torch', 'rb'))
        self.mu = t[0]
        self.std = t[1]
        # print(self.mu)
        # print(self.std)
        # number of previous states used for the prediction
        # self.history = 20
        self.past_sensors = []
        self.past_command = [np.zeros((48))]
        self.past_command[-1][0] = (1 - self.mu.data[0]) / self.std.data[0]
        self.past_command[-1][2] = (1 - self.mu.data[2]) / self.std.data[2]
        self.it = 0
        self.outCounter = 0
        self.stuckCounter = 0
        # self.input_dimensions = [3]+[i for i in range(5, 48)]

        if len(lst) > 3:
            self.history = lst[3]
        self.use_lstm = False
        self.use_lstm3 = False
        self.hn = Variable(torch.zeros(1, 256))
        self.cn = Variable(torch.zeros(1, 256))
        self.index = {}
        # names = ['accel', 'brake', 'gear', 'steer', 'clutch']
        # for i, x in enumerate(self.output_dimensions):
        #     self.index[names[x]] = i
        # self.output_dimensions = [3]  # [0, 1, 2, 3, 4]
        # self.state_dimensions = [i for i in range(5, 48)]
        self.past_command[-1] = self.past_command[-1][self.input_dimensions]
        print(time.time() - start)
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


    def carstate_matrix2(self, carstate):
        m = np.zeros((48))
        # m[0] = self.past_command[-1][0]
        # m[1] = self.past_command[-1][1]
        # m[2] = self.past_command[-1][2]
        # m[3] = self.past_command[-1][3]
        # m[4] = self.past_command[-1][4]

        DEGREE_PER_RADIANS = 180 / math.pi
        MPS_PER_KMH = 1000 / 3600

        m[5] = 0
        m[6] = carstate.angle / DEGREE_PER_RADIANS
        m[7] = carstate.current_lap_time
        m[8] = carstate.damage
        m[9] = carstate.distance_from_start
        m[10] = carstate.distance_raced
        m[11] = carstate.fuel
        m[12] = carstate.last_lap_time
        m[13] = carstate.race_position
        # for i in range(42, 47):
        #     m[i] = carstate.opponents[i-42]
        m[14] = carstate.rpm
        m[15] = carstate.speed_x / MPS_PER_KMH
        m[16] = carstate.speed_y / MPS_PER_KMH
        m[17] = carstate.speed_z / MPS_PER_KMH
        for i in range(18, 37):
            m[i] = carstate.distances_from_edge[i-18]
        m[37] = carstate.distance_from_center
        for i in range(38, 42):
            m[i] = carstate.wheel_velocities[i-38] / DEGREE_PER_RADIANS
        m[42] = carstate.z
        for i in range(43, 48):
            m[i] = carstate.focused_distances_from_edge[i-43]

        # return m[5:]
        return m[self.state_dimensions]


    def drive(self, carstate: State) -> Command:
        """
        # Produces driving command in response to newly received car state.
        #
        # This is a dummy driving routine, very dumb and not really considering a
        # lot of inputs. But it will get the car (if not disturbed by other
        # drivers) successfully driven along the race track.
        # """
        start = time.time()
        command = Command()
        # self.steer(carstate, 0.0, command)
        self.it += 1
        O = len(self.output_dimensions)
        #multiply speeds by 3.6
        # wheel velocities
        # features = self.carstate_matrix2(carstate)[[x - 5 for x in self.state_dimensions]].reshape(1,-1)
        features = self.carstate_matrix2(carstate).reshape(1,-1)
        # _logger.info(carstate)

        features = Variable(torch.FloatTensor(features))
        # features = Variable(torch.FloatTensor(features))

        t_features = self.model3.transform(features, self.mu[torch.LongTensor(self.state_dimensions)],
                                              self.std[torch.LongTensor(self.state_dimensions)])
        # if features[0, 18].data[0] == -1:
        #     _logger.info('im out')
            # t_features.data[0, 18:37] = 0
        outCommand = Command()
        if carstate.distances_from_edge[0] == -1:
            outCommand.meta = 1
        if len(self.past_command) >= self.history:

            feat2 = t_features.data
            if not self.use_lstm:
                for i in reversed(range(1, self.history)):
                    feat2 = torch.cat((feat2, torch.FloatTensor(self.past_command[-i]).view(1, -1)), 1)
            feat2 = Variable(feat2)


            if not self.use_lstm3:
                t_prediction = self.model3(feat2)
            else:
                # t_prediction, self.hn, self.cn = self.model.forward(feat2, self.hn, self.cn)
                t_prediction, self.hn = self.model3.forward(t_features, self.hn)

            # if not self.use_lstm:
            #     t_prediction0 = self.model0(feat2)
            # else:
            #     # t_prediction, self.hn, self.cn = self.model.forward(feat2, self.hn, self.cn)
            #     t_prediction0, self.hn = self.model0.forward(feat2, self.hn)
            #
            # t_prediction1 = self.model1(feat2)

            prediction = self.model3.back_transform(t_prediction,
                                                     self.mu[torch.LongTensor([3])],
                                                     self.std[torch.LongTensor([3])])

            # print(self.mu[torch.LongTensor([3])], self.std[torch.LongTensor([3])])
            # self.output_dimensions2 = [1]
            # prediction1 = self.model1.back_transform(t_prediction1,
            #                                        self.mu[torch.LongTensor(self.output_dimensions2)],
            #                                        self.std[torch.LongTensor(self.output_dimensions2)])

            # prediction0 = self.model0.back_transform(t_prediction0,
            #                                        self.mu[torch.LongTensor([0])],
            #                                        self.std[torch.LongTensor([0])])
            # # prediction3 = self.models[3].back_transform(t_prediction3,
            # #                                            self.mu[torch.LongTensor(self.output_dimensions)],
            # #                                            self.std[torch.LongTensor(self.output_dimensions)])

            prediction = prediction[0]
            # prediction1 = prediction1[0]
            # prediction0 = prediction0[0]
            # prediction3 = prediction3[0]

            # prediction.data[4] = min(1, prediction.data[4])
            # prediction.data[2] = np.rint(prediction.data[2])
            # prediction.data[4] = max(0, prediction.data[4])
            # prediction.data[2] = max(0, prediction.data[2])
            # prediction2.data[0] = min(1, prediction2.data[0])
            # outCommand.accelerator = prediction.data[0]
            # if prediction.data[0] > 0.005 or prediction.data[0] < -0.005:
            # _logger.info(prediction1.data[0])
            # _logger.info(carstate.speed_x)

            # _logger.info(prediction3.data[0])
            # print('we')
            if self.it > 10:
                # if carstate.speed_x < 200:
                #     outCommand.accelerator = prediction0.data[0]
                # if carstate.speed_x < 10:
                #     outCommand.accelerator = 1
                # if prediction.data[0] > 0.1 or prediction.data[0] < -0.1:
                self.accelerate(carstate, 50, outCommand)
                # else:
                #     self.accelerate(carstate, 400, outCommand)

                # outCommand.gear = 1
                # if carstate.distance_from_center > 0.8 or carstate.distance_from_center < 0.8:
                #     self.steer(carstate, 0.0, outCommand)
                # else:
                # if carstate.distance_from_center > 0.9 or carstate.distance_from_center < -0.9:
                #     self.outCounter += 1
                #     if self.outCounter > 100:
                #         self.outCounter = 0
                #         self.steer(carstate, 0.0, outCommand)
                # else:
                outCommand.steering = prediction.data[0]
                # self.steer(carstate, 0.0, outCommand)
                # if carstate.speed_x > 35:
                # outCommand.brake = prediction1.data[0]
                # else:
                #     print('other')
                #     outCommand.brake = prediction1.data[0] / 10
                # if outCommand.brake > 0.1:
                #     outCommand.accelerator = 0
                outCommand.clutch = 0#prediction.data[4]

            else:
                outCommand.accelerator = 1
                outCommand.gear = 1
                outCommand.steering = 0
                outCommand.brake = 0
                outCommand.clutch = 0

            t_prediction = self.model3.transform(prediction.view(1, -1),
                                              self.mu[torch.LongTensor(self.output_dimensions)],
                                              self.std[torch.LongTensor(self.output_dimensions)])[0]

            # _logger.info(prediction)
            # if prediction.data[2] < -2:
            #     print("we2")
            # _logger.info(carstate)
            # _logger.info(self.it)
            # t_features[0, :5] = prediction[:5]
        if self.input_dimensions != self.state_dimensions:
            t_features_numpy = np.hstack((np.zeros((1, O)), t_features.data.numpy()))
            t_features_numpy[0, :O] = t_prediction.data.numpy()[:O]
        else:
            t_features_numpy = t_features.data.numpy()
        self.past_command.append(t_features_numpy[0, :])
        # else:
        #     if self.input_dimensions != self.state_dimensions:
        #         numpy_feat = np.hstack((np.zeros((1, O)), t_features.data.numpy()))
        #         numpy_feat[0, :O] = self.past_command[-1][:O]
        #     else:
        #         numpy_feat = t_features.data.numpy()
        #     self.past_command.append(numpy_feat[0, :])


        # if self.data_logger:
        #         self.data_logger.log(carstate, command)

        # print(len(self.past_command))
        if len(self.past_command) > self.history:
            self.past_command = self.past_command[1:]
        print(time.time() - start, self.it)
        return outCommand

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        acceleration = math.pow(acceleration, 3)
        # acceleration = command.accelerator
        if acceleration > 0:

            if abs(carstate.distance_from_center) >= 1:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)
            #
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

    def on_restart(self):
        print("restarted")
        # self.__init__(logdata=False)
        pass