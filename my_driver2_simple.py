from pytocl.driver import Driver
from pytocl.car import State, Command,MPS_PER_KMH
import numpy as np
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
class MyDriver(Driver):
    # Override the drive method to create your own driver

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
        print("mehmehmeh")
        self.flag=False
    def drive(self, carstate: State):
        X=np.zeros(58)
        # # print("mehmeh",carstate)
        # # X[0]=carstate.speed

        X[0]=carstate.speed_x/220







        X[1]=carstate.angle
        # X[3]=carstate.distance_from_start
        X[3]=1 if carstate.distances_from_edge[0]<1 else 0
        X[4]=1 if carstate.distances_from_edge[1]<1 else 0
        X[5]=1 if carstate.distances_from_edge[2]<1 else 0
        X[6]=1 if carstate.distances_from_edge[3]<1 else 0
        X[7]=1 if carstate.distances_from_edge[4]<1 else 0
        X[8]=1 if carstate.distances_from_edge[5]<1 else 0
        X[9]=1 if carstate.distances_from_edge[6]<1 else 0
        X[10]=1 if carstate.distances_from_edge[7]<1 else 0
        X[11]=1 if carstate.distances_from_edge[8]<1 else 0
        X[12]=1 if carstate.distances_from_edge[9]<1 else 0
        X[13]=1 if carstate.distances_from_edge[10]<1 else 0
        X[14]=1 if carstate.distances_from_edge[11]<1 else 0
        X[15]=1 if carstate.distances_from_edge[12]<1 else 0
        X[16]=1 if carstate.distances_from_edge[13]<1 else 0
        X[17]=1 if carstate.distances_from_edge[14]<1 else 0
        X[18]=1 if carstate.distances_from_edge[15]<1 else 0
        X[19]=1 if carstate.distances_from_edge[16]<1 else 0
        X[20]=1 if carstate.distances_from_edge[17]<1 else 0
        X[21]=1 if carstate.distances_from_edge[18]<1 else 0
        # print(carstate.distances_from_edge[19])
        X[2]=carstate.distance_from_center
        X[22]=1 if carstate.opponents[0]<0.7 else 0
        X[23]=1 if carstate.opponents[1]<0.7 else 0
        X[24]=1 if carstate.opponents[2]<0.7 else 0
        X[25]=1 if carstate.opponents[3]<0.7 else 0
        X[26]=1 if carstate.opponents[4]<0.7 else 0
        X[27]=1 if carstate.opponents[5]<0.7 else 0
        X[28]=1 if carstate.opponents[6]<0.7 else 0
        X[29]=1 if carstate.opponents[7]<0.7 else 0
        X[30]=1 if carstate.opponents[8]<0.7 else 0
        X[31]=1 if carstate.opponents[9]<0.7 else 0
        X[32]=1 if carstate.opponents[10]<0.7 else 0
        X[33]=1 if carstate.opponents[11]<0.7 else 0
        X[34]=1 if carstate.opponents[12]<0.7 else 0
        X[35]=1 if carstate.opponents[13]<0.7 else 0
        X[36]=1 if carstate.opponents[14]<0.7 else 0
        X[37]=1 if carstate.opponents[15]<0.7 else 0
        X[38]=1 if carstate.opponents[16]<0.7 else 0
        X[39]=1 if carstate.opponents[17]<0.7 else 0
        X[40]=1 if carstate.opponents[18]<0.7 else 0
        X[41]=1 if carstate.opponents[19]<0.7 else 0
        X[42]=1 if carstate.opponents[20]<0.7 else 0
        X[43]=1 if carstate.opponents[21]<0.7 else 0
        X[44]=1 if carstate.opponents[22]<0.7 else 0
        X[45]=1 if carstate.opponents[23]<0.7 else 0
        X[46]=1 if carstate.opponents[24]<0.7 else 0
        X[47]=1 if carstate.opponents[25]<0.7 else 0
        X[48]=1 if carstate.opponents[26]<0.7 else 0
        X[49]=1 if carstate.opponents[27]<0.7 else 0
        X[50]=1 if carstate.opponents[28]<0.7 else 0
        X[51]=1 if carstate.opponents[29]<0.7 else 0
        X[52]=1 if carstate.opponents[30]<0.7 else 0
        X[53]=1 if carstate.opponents[31]<0.7 else 0
        X[54]=1 if carstate.opponents[32]<0.7 else 0
        X[55]=1 if carstate.opponents[33]<0.7 else 0
        X[56]=1 if carstate.opponents[34]<0.7 else 0
        X[57]=1 if carstate.opponents[35]<0.7 else 0


        # print(len(carstate.distances_from_edge))
        # print(X[3:22])

        summ=np.zeros(7)
        summop=np.zeros(7)
        for i in range(len(X[4:])):
            # print(len(X[4:]))
            if i <4 :
                summ[0]+=X[i]
                # summop[0]+=X[i]
            elif i<7:
                summ[1]+=X[i]
            elif i<9:
                summ[2]+=X[i]
            elif i==9:
                summ[3]+=X[i]
            elif i<12:
                summ[4]+=X[i]
            elif i<15:
                summ[5]+=X[i]
            elif i<19:
                summ[6]+=X[i]
            elif i<25:
                summop[0]+=X[i]
            elif i<31:
                summop[1]+=X[i]
            elif i<37:
                summop[2]+=X[i]
            elif i<43:
                summop[3]+=X[i]
            elif i<49:
                summop[4]+=X[i]
            elif i<55:
                summop[5]+=X[i]
            elif i<61:
                summop[6]+=X[i]
        averages=np.zeros(15)
        averages[0]=summ[0]/4
        averages[1]=summ[1]/3
        averages[2]=summ[2]/2
        averages[3]=summ[3]/2
        averages[4]=summ[4]/3
        averages[5]=summ[5]/4
        averages[6]=X[0]
        averages[7]=X[1]
        averages[8]=X[2]
        averages[9]=summop[0]/4
        averages[10]=summop[1]/3
        averages[11]=summop[2]/2
        averages[12]=summop[3]/2
        averages[13]=summop[4]/3
        averages[14]=summop[5]/4
        # print(averages)
        command = Command()
        action = self.net.advance(averages,0.002,0.002)
        # fitnesses = []
        sumofSensors=0
        for i in range(19):
            sumofSensors+=np.sqrt(carstate.distances_from_edge[i])

        # print((summ*0.3)+50)
        sensOff=0
        for i in range(19):
            if i<10:
                sensOff+=np.sqrt(carstate.distances_from_edge[i])
            else:
                sensOff-=np.sqrt(carstate.distances_from_edge[i])


        front=((carstate.distances_from_edge[8]+carstate.distances_from_edge[9])/2)
        brakeoff=0
        if front<100 :
            manualSpeed=70
            if not self.flag:
                brakeOff=0.50
                self.flag=True
            else:
                brakeOff=0

        else:
            self.flag=False
            brakeOff=0
            manualSpeed=front+50
        # print(front)
        # print(action)
        # print("heeeey")



        # if carstate.gear ==3:
        # 	gear=3
        # v_x= action[1]*240+0#(sumofSensors*1.3)#

        # self.accelerate(carstate,v_x,command)
        # self.steer(carstate,(action[0]*2)-1,command)
        # print(command.accelerator)

        gear=carstate.gear
        # X[0]=carstate.
        # X[0]=carstate.speed
# gear=carstate.gear
        inp=np.zeros(3)
        inp[0]=sensOff #command.accelerator
        # inp[1]=X[1]
        # inp[2]=X[2]
        # # X[0]=carstate.
        # X[0]=carstate.speed

        Kp=0.003
        dfromC=carstate.distance_from_center
        # if carstate.distance_from_center==-1:
        # 	dfromC=-10
        # elif carstate.distance_from_center==1:
        # 	dfromC=10
        error=(carstate.angle-dfromC*12)
        pidoff=Kp*carstate.angle*100
        pid=Kp*error
        # print(action)
        # print(pid)
        # X=list(carstate)
        # print(X)
        # self.net.Input(X)
        # self.net.Activate()
        # action = self.net.Output()



        # steeraction=action.index(max(action))
        # if steeraction==0:
        # 	command.steering=-0.7
        # elif steeraction==1:
        # 	command.steering=-0.4
        # elif steeraction==2:
        # 	command.steering=-0.2
        # elif steeraction==3:
        # 	command.steering=-0.1
        # elif steeraction==4:
        # 	command.steering=-0.05
        # elif steeraction==5:
        # 	command.steering=0
        # elif steeraction==6:
        # 	command.steering=0.05
        # elif steeraction==7:
        # 	command.steering=0.1
        # elif steeraction==8:
        # 	command.steering=0.2
        # elif steeraction==9:
        # 	command.steering=0.4
        # elif steeraction==10:
        # 	command.steering=0.7


        # # # print(action,"mehblah")
        if carstate.rpm>7000:
            gear+=1
        elif carstate.gear==0:
            gear+=1
        elif carstate.rpm<3000 and X[0]*220*3.6>10:
            gear-=1
        # # # if y[0][0] > 0:
        # ((action[0]*2)-1)*0.7
        # x=np.array(())
        # self.steer(carstate, ((action[0]*2)-1)*0.7, command)
  #       # else:
  #       #     command.brake = min(-acceleration, 1)
        # steer=action[0]
        # if (action[0]*2)-1<-0.2:
        # 	steer=-0.2
        # elif (action[0])*2-1>0.2:
        # 	steer=0.2
        # brake=0.01
        # if action[1]>0.8:
        # 	accel=0.8
        # elif action[1]<0.4:
        # 	accel=0.4


        # brake=action[2]
        # if action[2]>0.2:
        # 	brake=0.2
        # elif action[1]<0.4:
        # 	accel=0.4

        # v_x = (((((action[2]-1)*-1)**2)-1)*-1*100)+95

        # if carstate.rpm < 2500:
        # 	command.gear = carstate.gear - 1

        # if not command.gear:
        # 	command.gear = 1
        # print("mehblah",carstate.distance_from_center)
        # self.data_logger.log(carstate,command)
        # print(y)
        # if carstate.distance_from_center>1.5 or carstate.distance_from_center<-1.5:

        # 	self.data_logger.close()
        # 	self.data_logger = None
        # else:
        # print(action)
                # if carstate.distances_from_edge[8]<65 and carstate.distances_from_edge[8]>0:
        # 	command.brake=(((carstate.distances_from_edge[8]/65)-1)*-1)**3
        # # if carstate.distances_from_edge[8]==-1.0:
        # 	command.brake=0
        # 	command.accelerator=0.8
        # 	command.steering=pidoff
        # print(action)
        # print((action[0]*2)-1)

        if action[1]<0.3:
            command.accelerator=1
            command.brake=0
        else:
            command.accelerator=0
            command.brake=action[1]
        # command.accelerator=1-action[1]#0.4#accel#y[0][0]
        # command.brake=action[1]#((abs(action[0]))-0.5)/5#abs(pid)*2#y[0][1]*y[0][1]
        command.steering=(action[0]*2)-1
        # command.steering=((action[0]*2)-1)*0.7#steer
        command.gear=gear
        command.focus=0
        # command.gear=1
        # print(command)
        # print(command)
        return command
