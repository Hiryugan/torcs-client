from pytocl.driver import Driver
from pytocl.car import State, Command,MPS_PER_KMH
import numpy as np
import math
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
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController

# loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
import logging

from pytocl.analysis import DataLogWriter
import pickle
class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    def __init__(self,log_data=False, net=None):
        # self.steering_ctrl = CompositeController(ProportionalController(0.4),IntegrationController(0.2, integral_limit=1.5),DerivativeController(2))
        # self.acceleration_ctrl = CompositeController(
        # 	ProportionalController(3.7),
        # )
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if log_data else None
        self.net=net
        self.flag=False

        # somehow parse
        t = pickle.load(open('pre_train/models/ustd_torch', 'rb'))
        # t = load_model(open('pre_train/models/ustd_torch', 'rb'))
        self.mu = t[0]
        self.std = t[1]



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

        return m[5:]
        # return m[self.state_dimensions]


    def drive(self, carstate: State):

        features = self.carstate_matrix2(carstate)[[x - 5 for x in self.state_dimensions]]
        # print(averages)
        command = Command()
        action = self.net.advance(features, 0.1, 5)



        sensOff=0
        for i in range(19):
            if i<10:
                sensOff+=np.sqrt(carstate.distances_from_edge[i])
            else:
                sensOff-=np.sqrt(carstate.distances_from_edge[i])



        self.accelerate(carstate,v_x,command)
        self.steer(carstate,(action[0]*2)-1,command)
        # print(command.accelerator)


        if carstate.rpm>7000:
            outCommand.gear+=1
        elif carstate.gear==0:
            outCommand.gear+=1
        elif carstate.rpm<2000 and X[0]>10:
            outCommand.gear-=1

        if command.accelerator==0:
            command.brake=0.02
        else:
            command.brake=0


        return command

