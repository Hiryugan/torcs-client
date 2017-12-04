from pytocl.driver import Driver
from pytocl.car import State, Command,MPS_PER_KMH
import numpy as np
import math
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
# from keras.models import model_from_json
# import keras
# with open('model.json', 'r') as json_file:
# 	loaded_model_json = json_file.read()
# json_file.close()	
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# # print("Loaded model from disk")
from pytocl.main import main
from config_parser import Configurator, Parser
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController

# loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
import logging

from pytocl.analysis import DataLogWriter
import pickle
class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    def __init__(self, parser, models=None, log_data=False, net=None):
        super(MyDriver, self).__init__()

        ## get filename in the arguments
        # self.parser = Parser(config)
        self.parser = parser
        self.net = None

        # self.models = {}
        self.o2i = {'accel': 0, 'brake': 1, 'steer': 2}
        self.i2o = {0: 'accel', 1: 'brake', 2: 'steer'}
        # print(self.parser.output_model)
        self.change_outputs = {}
        if models is not None:
            self.models = models
        # else:
        for i in range(len(self.parser.output_model)):
        #     print(self.parser.output_model[i])
        #     if self.parser.output_model[i]['name'] != 'None':
            #     self.models[self.i2o[i]] = pickle.load(open(self.parser.output_model[i]['name'], 'rb'))

            if self.parser.output_model[i]['used'] == True:
                self.change_outputs[self.i2o[i]] = True
            else:
                self.change_outputs[self.i2o[i]] = False

        self.use_supervised = False
        if len(self.models) > 0:
            self.model = self.models['steer']
            self.use_supervised = True
            self.history = self.models['steer'].history_size
            self.input_dimensions = self.models['steer'].input_dimensions
            self.output_dimensions = self.models['steer'].output_dimensions
            self.state_dimensions = self.models['steer'].state_dimensions
        self.state_dimensions = [6, 15, 16, 17] + list(range(18, 38)) + [42]

            # prefix_folder = self.parser.output_model['steer'].split('/')[:-1]

            # t = pickle.load(open(prefix_folder+'/ustd_torch', 'rb'))
            # self.mu = t[0]
            # self.std = t[1]

        self.past_command = [np.zeros((48))]
        self.it = 0


        self.data_logger = DataLogWriter() if log_data else None
        self.net=net
        self.flag=False

        # somehow parse
        #



    def carstate_matrix2(self, carstate):
        m = np.zeros((48))

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


    def drive(self, carstate: State):

        features = self.carstate_matrix2(carstate).reshape(1, -1)

        outCommand = Command()
        predictions = {}
        if self.use_supervised:
            features = Variable(torch.FloatTensor(features))

            t_features = self.model.transform(features, self.model.mu[torch.LongTensor(self.state_dimensions)],
                                              self.model.std[torch.LongTensor(self.state_dimensions)])


            if len(self.past_command) >= self.history:

                feat2 = t_features.data
                for i in reversed(range(1, self.history)):
                    feat2 = torch.cat((feat2, torch.FloatTensor(self.past_command[-i]).view(1, -1)), 1)
                feat2 = Variable(feat2)

                for key, model in self.models.items():


                        t_prediction = model(feat2)

                        prediction = model.back_transform(t_prediction,
                                                                model.mu[torch.LongTensor([i])],
                                                                model.std[torch.LongTensor([i])])
                        # reshape to array
                        prediction = prediction[0]
                        predictions[key] = prediction.data.numpy()[0]

                        # if self.parser.output_model[self.i2o[i]]['usage'] == True:
                        #     features = np.hstack((features, prediction.data.numpy()))





                # instead stack to t_features
            t_features_numpy = t_features.data.numpy()

            # done in any case - both beginning and with timestep > history
            self.past_command.append(t_features_numpy[0, :])

            # features = np.hstack((features, prediction))
            if len(self.past_command) > self.history:
                for key, model in self.models.items():
                    if self.change_outputs[key] == True:
                        t_features_numpy = np.hstack((t_features_numpy, predictions[key].reshape(1, -1)))

            if len(self.past_command) > self.history:
                self.past_command = self.past_command[1:]
        # if not using supervised learning
        else:
            t_features_numpy = features

        # genetic_prediction = self.net.advance(features[0], 0.1, 5)

        # self.accelerate(carstate, v_x, outCommand)
        # self.steer(carstate, (genetic_prediction[0] * 2) - 1, outCommand)

        if self.it > 20:
            genetic_prediction = self.net.activate(t_features_numpy[0])

            outCommand.clutch = 0  # prediction.data[4]
            # print(genetic_prediction)
            # if genetic_prediction[0] > 0:
            #     if carstate.rpm > 8000:
            #         outCommand.gear = carstate.gear + 1
            # if carstate.rpm < 2500:
            #     outCommand.gear = carstate.gear - 1
            # if not outCommand.gear:
            #     outCommand.gear = carstate.gear or 1

            if carstate.rpm > 7000:
                outCommand.gear = carstate.gear + 1
            elif carstate.gear <= 0:
                outCommand.gear = carstate.gear + 1
            elif carstate.rpm < 2500 and carstate.gear > 1:
                outCommand.gear = carstate.gear - 1
            if not outCommand.gear:
                outCommand.gear = carstate.gear or 1
            # if outCommand.gear <= 0:
            #     outCommand.gear += 1
            # if carstate.rpm > 8000:
            #     outCommand.gear += 1
            # elif carstate.gear == 0:
            #     outCommand.gear += 1
            # elif carstate.rpm < 2000:
            #     outCommand.gear -= 1


            idx = 0
            if self.parser.merge_accel_brake == True:
                if genetic_prediction[idx] > 0:
                    outCommand.accelerator = genetic_prediction[idx]
                    outCommand.brake = 0
                else:
                    outCommand.accelerator = 0
                    outCommand.brake = -genetic_prediction[idx]
                idx += 1
            else:
                if 'accel' in self.models:
                    # if not self.change_outputs['accel']:
                    outCommand.accelerator = predictions['accel']
                else:
                    outCommand.accelerator = genetic_prediction[idx]
                    idx += 1
                if 'brake' in self.models:
                    # if not self.change_outputs['brake']:
                    outCommand.brake = predictions['brake']

                else:
                    outCommand.brake = genetic_prediction[idx]
                    idx += 1
            if 'steer' in self.models:
                if not self.change_outputs['steer']:
                    outCommand.steering = predictions['steer']
                else:
                    outCommand.steering = genetic_prediction[idx]
                    idx += 1

            if self.parser.avoid_go_out == True:
                if carstate.distance_from_center < -0.8 or carstate.distance_from_center > 0.8:
                    self.steer(carstate, 0.0, outCommand)
        else:

            outCommand.accelerator = 1
            outCommand.gear = 1
            outCommand.steering = 0
            outCommand.brake = 0
            outCommand.clutch = 0

        # self.accelerate(carstate, 50, outCommand)
        # print(outCommand)
        self.it += 1

        return outCommand

    def set_net(self, net):
        self.net = net

    def on_restart(self):
        # print("restarted")
        self.it = 0

        # self.__init__(logdata=False)

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )

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
