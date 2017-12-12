import logging
import os
import marshal
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
import copy
import numpy as np

from pre_train.mlp_torch_2 import MLP#, transform, back_transform, load_model
import torch.nn.functional as F

# from .pre_train.mlp_torch import MLP, transform, back_transform
# _logger = logging.getLogger(__name__)

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
        # self.data_logger = DataLogWriter() if logdata else None
        # lst = pickle.load(open('pre_train/models/dimensions', 'rb'))
        # self.output_dimensions = [3]#lst[0]
        self.data = []
        self.brake = 0
        self.models = dict()
        # for i in [0,3]:
        #     self.models[i] = pickle.load(open('pre_train/models/mod_temporal_torch' + '_' + str(i),'rb'))
        self.model3 = torch.load('pre_train/models/mod_temporal_torch_3', lambda storage, location: storage)
        # self.model3 = self.model3.cpu()
        # self.model1 = pickle.load(open('pre_train/models/mod_temporal_torch_1','rb'))
        # self.modelv = torch.load('pre_train/models/mod_temporal_torch_15.torch', lambda storage, location: storage)
        # self.model0 = pickle.load(open('pre_train/models/mod_temporal_torch_0','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        # self.model = load_model(open('pre_train/models/mod_temporal_torch','rb'))
        # self.logger = open('logger', 'w')
        # t = pickle.load(open('pre_train/models/ustd_torch', 'rb'))
        # t = load_model(open('pre_train/models/ustd_torch', 'rb'))
        self.mu = self.model3.mu
        self.std = self.model3.std
        # print(self.mu)
        # print(self.std)
        # number of previous states used for the prediction
        # self.history = 20
        self.past_sensors = []
        self.past_command = [np.zeros((len(self.model3.state_dimensions)))]
        # self.past_command2 = [np.zeros((len(self.modelv.state_dimensions)))]
        # self.past_command[-1][0] = (1 - self.mu.data[0]) / self.std.data[0]
        # self.past_command[-1][2] = (1 - self.mu.data[2]) / self.std.data[2]
        self.it = 0
        self.outCounter = 0
        self.stuckCounter = 0
        self.use_default = 0
        # self.input_dimensions = [3]+[i for i in range(5, 48)]
        self.state_dimensions = self.model3.state_dimensions
        self.input_dimensions = self.model3.input_dimensions
        # if len(lst) > 3:
        #     self.history = lst[3]
        self.history = self.model3.history_size
        self.use_lstm = False
        self.use_lstm3 = False
        self.min_dist = 0
        self.hn = Variable(torch.zeros(1, 256))
        self.cn = Variable(torch.zeros(1, 256))
        self.index = {}
        # names = ['accel', 'brake', 'gear', 'steer', 'clutch']
        # for i, x in enumerate(self.output_dimensions):
        #     self.index[names[x]] = i
        # self.output_dimensions = [3]  # [0, 1, 2, 3, 4]
        # self.state_dimensions = [i for i in range(5, 48)]
        # self.past_command[-1] = self.past_command[-1][self.model3.state_dimensions]
        # self.past_command2[-1] = self.past_command2[-1][self.modelv.state_dimensions]
        print(time.time() - start)
        self.back_stuck = 0
        self.front_stuck = 0
        self.use_default = 0
        self.just_go = 0
        """ ******************************************* """
        """ swarm stupidity """
        """ ******************************************* """
        self.filename1 = 'swarm_file1.marshal'
        self.filename2 = 'swarm_file2.marshal'
        self.filename3 = 'deleted.txt'
        self.dict1 = {}
        self.dict2 = {}
        self.pause_swarm = 0
        self.wait_time = 0
        self.use_swarm_at_all = True

        if os.path.isfile(self.filename1):
            self.swarm_file1 = open(self.filename2, "wb", os.O_NONBLOCK)
            # self.swarm_file1 = open(self.filename2, "wb+")
        else:
            self.swarm_file1 = open(self.filename1, "wb", os.O_NONBLOCK)

        """ ******************************************* """
        """ swarm stupidity """
        """ ******************************************* """
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
        # if self.data_logger:
        #     self.data_logger.close()
        #     self.data_logger = None
        pass

    def carstate_matrix2(self, carstate):

        m = np.zeros((85))
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
        for i in range(48, 84):
                m[i] = 200 - carstate.opponents[i-48]
        return m

    def drive(self, carstate: State) -> Command:
        """
        # Produces driving command in response to newly received car state.
        #
        # This is a dummy driving routine, very dumb and not really considering a
        # lot of inputs. But it will get the car (if not disturbed by other
        # drivers) successfully driven along the race track.
        # """
        start = time.time()
        carstate_orig = copy.deepcopy(carstate)

        self.data.append(self.carstate_matrix2(carstate))

        """ ******************************************* """
        """ swarm stupidity """
        """ ******************************************* """
        if self.pause_swarm > 0:
            self.pause_swarm -= 1

        if self.it == 0:
            self.dict1 = {}
            self.dict2 = {}
            if self.swarm_file1.name == self.filename1:
                try:
                    self.swarm_file2 = open(self.filename2, 'rb+')
                except:
                    self.use_swarm_at_all = False
                    if os.path.isfile(self.filename1):
                        os.remove(self.filename1)
                    if os.path.isfile(self.filename2):
                        os.remove(self.filename2)

            else:
                try:
                    self.swarm_file2 = open(self.filename1, 'rb+')
                except:
                    self.use_swarm_at_all = False
                    if os.path.isfile(self.filename1):
                        os.remove(self.filename1)
                    if os.path.isfile(self.filename2):
                        os.remove(self.filename2)

            if self.use_swarm_at_all:
                print("file1: ", self.swarm_file1.name)
                print('file2: ', self.swarm_file2.name)

            if os.path.isfile(self.filename3):
                os.remove(self.filename3)

        if self.use_swarm_at_all:
            if os.stat(self.swarm_file2.name).st_size > 0:
                try:
                    self.swarm_file2.seek(0)
                    tup2 = marshal.load(self.swarm_file2)
                    self.dict2[tup2[0]] = (tup2[1], tup2[2])
                    self.swarm_file2.seek(0)
                except Exception as ex:
                    print(ex)

        """ ******************************************* """
        """ swarm stupidity """
        """ ******************************************* """
        """ ******************************************* """
        """ unstucking """
        """ ******************************************* """
        start = time.time()
        # self.model3.eval()
        outCommand = Command()

        if self.just_go > 0:
            self.just_go -= 1

        print("use_default: ", self.use_default, "just_go: ", self.just_go)
        if carstate.current_lap_time > 5 and self.just_go == 0:
            self.update_stuck(carstate)
            if self.use_default > 0:
                self.use_default -= 1
                self.accelerate(carstate, 20, outCommand)
                self.steer(carstate, 0.0, outCommand)
                if self.use_default == 0:  # and self.:
                    self.just_go = 200
                    self.pause_swarm = 400
                    outCommand.clutch = 0
                    outCommand.gear = 0
                    outCommand.accelerator = 1
                    outCommand.brake = 0

                return outCommand

            if self.front_stuck > 30:
                # if self.front_stuck > 15:
                #     self.min_dist = min(carstate.distances_from_edge)
                outCommand.accelerator = max(0, min(1, 1.3 - 0.7 * self.min_dist))
                # outCommand.accelerator = 0.5
                outCommand.gear = -1
                outCommand.brake = 0.0
                outCommand.clutch = 0.0
                outCommand.steering = -1 * carstate.angle * np.pi / (180.0 * 0.785398)
                return outCommand

            # else:
            elif self.back_stuck > 15:
                outCommand.accelerator = 1
                outCommand.gear = 1 if carstate.gear <= 0 else carstate.gear
                outCommand.brake = 0.0
                outCommand.clutch = 0.0
                outCommand.steering = carstate.angle * np.pi / (180.0 * 0.785398)
                outCommand.steering -= 0.35 * np.sign(carstate.distance_from_center) * min(1.5, math.fabs(
                    carstate.distance_from_center))
                return outCommand
        """ ******************************************* """
        """ unstucking """
        """ ******************************************* """

        start = time.time()
        # self.steer(carstate, 0.0, command)
        self.it += 1
        # features = self.carstate_matrix2(carstate)[[x - 5 for x in self.state_dimensions]].reshape(1,-1)
        features = self.carstate_matrix2(carstate)[self.model3.state_dimensions].reshape(1,-1)
        # _logger.info(carstate)
        # featuresv = self.carstate_matrix2(carstate)[self.modelv.state_dimensions].reshape(1, -1)
        features = Variable(torch.FloatTensor(features), requires_grad=False)
        # features = Variable(torch.FloatTensor(features))
        # featuresv = Variable(torch.FloatTensor(featuresv))
        startt = time.time()
        t_features = self.model3.transform(features, self.model3.mu[torch.LongTensor(self.model3.state_dimensions)],
                                              self.model3.std[torch.LongTensor(self.model3.state_dimensions)])

        # t_featuresv = self.modelv.transform(featuresv, self.modelv.mu[torch.LongTensor(self.modelv.state_dimensions)],
        #                                    self.modelv.std[torch.LongTensor(self.modelv.state_dimensions)])
        # print('transofrms', time.time() - startt)
        # print(torch.sum(self.model3.std), torch.sum(self.model3.mu))
        # print(torch.sum(self.model1.std), torch.sum(self.model1.mu))
        # if features[0, 18].data[0] == -1:
        #     _logger.info('im out')
            # t_features.data[0, 18:37] = 0

        # if carstate.distances_from_edge[0] == -1:
        #     outCommand.meta = 1
        if len(self.past_command) >= self.history:

            feat2 = t_features.data
            for i in reversed(range(1, self.model3.history_size)):
                feat2 = torch.cat((feat2, torch.FloatTensor(self.past_command[-i]).view(1, -1)), 1)
            feat2 = Variable(feat2, requires_grad=False)

            # featv = t_featuresv.data
            # for i in reversed(range(1, self.modelv.history_size)):
            #     featv = torch.cat((featv, torch.FloatTensor(self.past_command2[-i]).view(1, -1)), 1)
            # featv = Variable(featv)

            t_prediction = self.model3(feat2)
            # if not self.use_lstm:
            #     t_prediction0 = self.model0(feat2)
            # else:
            #     # t_prediction, self.hn, self.cn = self.model.forward(feat2, self.hn, self.cn)
            #     t_prediction0, self.hn = self.model0.forward(feat2, self.hn)
            #
            # t_predictionv = self.modelv(featv)
            # s2 = time.time()
            prediction = self.model3.back_transform(t_prediction,
                                                     self.model3.mu[torch.LongTensor(self.model3.output_dimensions)],
                                                     self.model3.std[torch.LongTensor(self.model3.output_dimensions)])
            # predictionv = self.modelv.back_transform(t_predictionv,
            #                                        self.modelv.mu[torch.LongTensor(self.modelv.output_dimensions)],
            #                                        self.modelv.std[torch.LongTensor(self.modelv.output_dimensions)])
            #

            prediction = prediction[0]
            # predictionv = predictionv[0]
            # prediction = F.tanh(prediction)
            # predictionv[0] = F.sigmoid(predictionv[0])
            # predictionv[1] = F.sigmoid(predictionv[1])
            # prediction[2] = F.tanh(prediction[2])
            # predictionv = predictionv[0]
            # predictionv = F.sigmoid(predictionv)

            # _logger.info(prediction3.data[0])
            # print('we')
            # print(a.data[0], b.data[0], s.data[0])

            """ ******************************************* """
            """ swarm stupidity with car2 waiting 10 seconds """
            """ ******************************************* """
            # if self.swarm_file1.name == self.filename2 and self.it < 50:
            #     self.wait_time = 500
            #
            # if self.swarm_file1.name == self.filename2 and self.wait_time > 0:
            #     self.wait_time -= 1
            #     outCommand.accelerator = 0
            #     outCommand.gear = 0
            #     outCommand.steering = 0
            #     outCommand.brake = 0
            #     outCommand.clutch = 0
            #     print("WAITING ", self.wait_time)
            #     return outCommand
            """ ******************************************* """
            """ swarm stupidity """
            """ ******************************************* """
            if self.it > 100:

                # print(predictionv.data[0])
                outCommand.steering = prediction.data[0]
                """ ******************************************* """
                """ swarm stupidity """
                """ ******************************************* """
                if self.use_swarm_at_all:
                    # steering_threshold = 0.05
                    # 0.5 is actually pretty hardcore steering
                    relative_dist = int(carstate.distance_from_start) + 30
                    swarm_speed = 0
                    if relative_dist in self.dict2 and self.just_go == 0 and self.use_default == 0:
                        steer2 = abs(self.dict2[relative_dist][0])
                        print("*" * 50)
                        print(relative_dist, " ", steer2, " count:", self.dict2[relative_dist][1])
                        if self.pause_swarm == 0:
                            swarm_speed = (1 - steer2 * 2) * 75 - 25
                        print("steer2: ", steer2, " swarm_speed: ", swarm_speed, "pause_swarm: ", self.pause_swarm)

                    dist1 = int(carstate.distance_from_start)
                    steer1 = outCommand.steering

                    if dist1 not in self.dict1:
                        self.dict1[dist1] = (steer1, 1)  # create new entry
                        tup1 = (dist1, steer1, 1)
                    else:
                        new_count = self.dict1[dist1][1] + 1
                        new_avg = ((self.dict1[dist1][0] * self.dict1[dist1][1]) + steer1) / new_count
                        self.dict1[dist1] = (new_avg, new_count)  # update dict
                        tup1 = (dist1, new_avg, new_count)

                    DEGREE_PER_RADIANS = 180 / math.pi
                    if swarm_speed > 0 and (abs(carstate.angle / DEGREE_PER_RADIANS) > 30):
                        swarm_speed = 0
                        print("!!!!!!!!!!!!!!!!\nPREVENTING swarm\n!!!!!!!!!!!!!!!")
                    print('we guido')
                try:
                    print('here we are!')
                    self.accelerate(carstate,
                                    (np.sum(carstate.distances_from_edge) / 600) ** 2 * (150 + swarm_speed) + 10,
                                    outCommand)
                except:
                    self.accelerate(carstate, (np.sum(carstate.distances_from_edge) / 600) ** 2 * 150 + 10, outCommand)
                """ ******************************************* """
                """ swarm stupidity """
                """ ******************************************* """
                # if carstate.distance_from_center > 0.9 or carstate.distance_from_center < -0.9:
                #     self.steer(carstate, 0.0, outCommand)
                #     _logger.info(prediction.data[0])

                # print(predictionv.data[0])
                # outCommand.accelerator = prediction.data[0]
                # self.change_gear(carstate, outCommand)
                # if predictionv.data[1] > 0.5:
                #     if self.brake == 0:
                #         outCommand.brake = predictionv.data[1]
                #         outCommand.accelerator = 0
                #         self.brake = 1
                #     else:
                #         self.brake = 0
                # else:
                #     outCommand.accelerator = predictionv.data[0]
                #     outCommand.brake = 0
                #     self.brake = 0
                # outCommand.steering = s
                # self.accelerate(carstate, (np.sum(carstate.distances_from_edge) / 600)**2 * 150 + 10, outCommand)
                # self.accelerate(carstate, predictionv.data[0], outCommand)
                # print(predictionv.data[0])

                # self.change_gear(carstate, outCommand)
                # if predictionv.data[0] > 0.05:
                #     outCommand.brake = 1
                #     outCommand.accelerator = 0
                # else:
                #     outCommand.accelerator = 1
                #     outCommand.brake = 0
                    # self.accelerate(carstate, predictionv.data[0], outCommand)


                outCommand.clutch = 0#prediction.data[4]

            else:
                outCommand.accelerator = 1
                outCommand.gear = 1
                outCommand.steering = 0
                outCommand.brake = 0
                outCommand.clutch = 0
                """ ******************************************* """
                """ swarm stupidity """
                """ ******************************************* """
                tup1 = (carstate.distance_from_start, 0, 1)
                """ ******************************************* """
                """ swarm stupidity """
                """ ******************************************* """
            outCommand.brake = 0

        t_features_numpy = t_features.data.numpy()
        # t_features_numpy1 = t_featuresv.data.numpy()
        self.past_command.append(t_features_numpy[0, :])
        # self.past_command2.append(t_features_numpy1[0, :])

        if carstate.distance_from_center > 0.99 or carstate.distance_from_center < -0.99:
            self.steer(carstate, 0.0, outCommand)
            self.accelerate(carstate, 20, outCommand)

        # if self.data_logger:
        #         self.data_logger.log(carstate, command)

        # print(len(self.past_command))
        if len(self.past_command) > self.history:
            self.past_command = self.past_command[1:]
        # if len(self.past_command2) > self.history:
        #     self.past_command2 = self.past_command2[1:]
        # print(time.time() - start, self.it)
        # print(time.time() - start)
        print(self.it, time.time() - start, outCommand.steering, carstate.distance_from_center, carstate.distance_from_start, carstate.current_lap_time)

        print(self.it, time.time() - start, outCommand.steering, carstate_orig.distance_from_center,
              carstate_orig.distance_from_start, carstate_orig.current_lap_time)
        """ ******************************************* """
        """ swarm stupidity """
        """ ******************************************* """

        if self.use_swarm_at_all:
            try:
                self.swarm_file1.seek(0)
                self.swarm_file1.truncate()
                if self.pause_swarm == 0:
                    marshal.dump(tup1, self.swarm_file1)
                    # print("dumping ", tup1[1])
                else:
                    print("NOT dumping")
                self.swarm_file1.flush()
                os.fsync(self.swarm_file1.fileno())
                # self.swarm_file1.seek(0)
            except Exception as ex:
                print(ex)
                print("couldnt dump")
        """ ******************************************* """
        """ swarm stupidity """
        """ ******************************************* """
        stop = time.time()
        print('Time passed: ', stop - start)
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
        if acceleration > 0.0:

            if abs(carstate.distance_from_center) >= 1:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        else:
            command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            # if carstate.gear == 0:
            #     if carstate.speed_x > 2:
            #         command.gear = carstate.gear - 1
            if carstate.gear > 0:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )

    def set_net(self, net):
        self.net = net

    def change_gear(self, carstate, command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        # else:
        #     command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            # if carstate.gear == 0:
            #     if carstate.speed_x > 2:
            #         command.gear = carstate.gear - 1
            if carstate.gear > 0:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        return command

    def on_restart(self):
        print("restarted")
        self.it = 0

    def update_stuck(self, carstate):
        self.min_dist = min(carstate.distances_from_edge)
        if carstate.speed_x / MPS_PER_KMH < 3 \
                and (math.fabs(carstate.distance_from_center) >= 0.93 or self.min_dist < 3) \
                and math.fabs(carstate.angle) > 15 \
                and carstate.angle * carstate.distance_from_center < 0:
            self.front_stuck += 1
            print(carstate.speed_x)
            print(carstate.speed_x / MPS_PER_KMH)
            print(self.front_stuck, '% getting front stuck 1')
        elif carstate.speed_x / MPS_PER_KMH < 3 and carstate.rpm > 5000 and carstate.gear >= 0:
            self.front_stuck += 1
            print(carstate.speed_x)
            print(carstate.speed_x / MPS_PER_KMH)
            print(self.front_stuck, '% getting front stuck 2')
        else:
            if self.front_stuck > 0 and self.use_default == 0:
                self.use_default = 50
            self.front_stuck = 0
            # self.front_stuck -= self.front_stuck % 3
            # self.front_stuck = max(0, self.front_stuck)

        if carstate.speed_x / MPS_PER_KMH < 8 \
                and carstate.angle * np.sign(carstate.distance_from_center) >= -15:
            # and math.fabs(carstate.angle) < 90 \
            self.back_stuck += 1
            print(carstate.speed_x)
            print(self.back_stuck, '% getting back stuck 1')
        else:
            if self.back_stuck > 0 and self.use_default == 0:
                self.use_default = 50
            self.back_stuck = 0
            # self.back_stuck -= self.front_stuck % 3
            # self.back_stuck = max(0, self.back_stuck)
# self.__init__(logdata=False)

