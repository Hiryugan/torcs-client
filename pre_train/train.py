from pre_train.mlp_torch_2 import MLP
from pre_train.mlp import MLP2
from pre_train.reader import construct_dataset_lstm, construct_dataset, construct_dataset_velocity, construct_dataset_dirt
from pre_train.lstm import myLSTM
import pickle
import torch
# indices of the input dimensions for the past frames
# notice, 48 uses all dimensions but the opponent sensors
# use the first 5 dimensions as outputs (accel, brake, gear, steer, clutch)
# output_dimensions = [0, 1, 2, 3, 4]
output_dimensions = [3]#[0, 1, 2, 3, 4]
# dimensions of the state input (all the car state but opponents)
# state_dimensions = [i for i in range(5, 48)]
# state_dimensions = [15] + list(range(18, 37))
state_dimensions = [6, 15, 16, 17] + list(range(18, 38)) + list(range(48, 84))
input_dimensions = state_dimensions
history_size = 4
batch_size = 512
cuda = True
epochs = 50
# todo: correct data with -1, avoid frames without change in angle

# dataset_names = [ 'aalborg', 'alpine', 'cg_speedway_number1', 'etrack3', 'spring']
# dataset_names = [ 'aalborg', 'alpine', 'cg_speedway_number1', 'spring']
# dataset_names = ['alpine', 'aalborg', 'forza', 'etrack3', 'etrack2', 'spring', 'wheel1', 'street1']
d2 = ['etrack3', 'oval3', 'street1', 'aalborg', 'alpine', 'wheel2', 'forza', 'etrack2']
# dataset_names = ['dirt1_2', 'dirt2_2', 'dirt3_2', 'dirt4_2', 'dirt5_2', 'dirt6_2']
dataset_names = ['aalborg_5-4_ok', 'alpine_5-3_ok', 'alpine_8-5_difficult', 'brondehach_8-6_ok', 'cg_speedway_ok', 'cg_track2_wiggle', 'eroad_8-3_ok',
                'etrack2_5-4_ok', 'etrack_8-3_bumps', 'forza_8-6_wiggle', 'olethros_road_ok', 'wheel2_6-2_3bumps']
dataset_names += ['etrack3_opp9','etrack2_opp4', 'wheel2_opp6',  'street1_opp10', 'aalborg_opp7','forza_opp4',  'eroad_opp3', 'corkscrew_opp1']
# dataset_names += ['etrack3_opp9','etrack2_opp4', 'wheel2_opp6',  'street1_opp10', 'aalborg_opp7','forza_opp4', 'spring_opp10', 'alpine_opp5', 'eroad_opp3', 'corkscrew_opp1']
# d2 = ['olethros_road_1', 'etrack4', 'etrack', 'cg_track_2', 'alpine2']
# d2 = ['olethros_road_1', 'forza', 'street1','etrack4', 'etrack', 'cg_track_2', 'aalborg' , 'alpine2']
    # ,'etrack3_opp9','etrack2_opp4', 'wheel2_opp6',  'street1_opp10']#, 'aalborg_opp7','forza_opp4', 'spring_opp10', 'alpine_opp5', 'eroad_opp3', 'corkscrew_opp1']
# dataset_names_test = ['dirt1','dirt2','dirt3','dirt4', 'dirt5','dirt6', 'forza', 'olethros_road_1', 'ruudskogen', 'street1', 'aalborg' , 'alpine2',  'corkscrew', 'eroad', 'etrack6']
# dataset_names_test = ['dirt1','dirt2','dirt3','dirt4','dirt6','mixed1','mixed2']
dataset_names_test = ['oval1']#,'oval2','oval3','oval4','oval5','oval6','oval7','oval9','oval10', ]
# dataset_names_test = ['etrack3', 'forza','cg_track3','etrack4', 'ruudskogen','etrack', 'cg_track_2','spring' , 'aalborg', 'alpine2', 'cg_speedway_number1', 'olethros_road_1', 'brondehach']
dataset_names2 = [x + '_2' for x in d2] + dataset_names
# dataset_names2 = [x  for x in dat aset_names]
# dataset_names2 = [x + for x in dataset_names]
dataset_names3 = [x + '_2' for x in dataset_names_test[-1:]]

# construct_fun = construct_dataset_lstm
# model = myLSTM(input_dimensions = input_dimensions,output_dimensions = output_dimensions, state_dimensions=state_dimensions, batch_size = batch_size, history_size = history_size, cuda = cuda, epochs = epochs)

construct_fun = construct_dataset
# construct_fun = construct_dataset_dirt
# construct_fun = construct_dataset_velocity
model = MLP2(input_dimensions = input_dimensions,output_dimensions = output_dimensions, state_dimensions=state_dimensions, batch_size = batch_size, history_size = history_size, cuda = cuda, epochs = epochs)
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3, normalize=True)
model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3, construct_dataset_function=construct_fun, normalize=True)

loss, mu, std = model.train()
for i in output_dimensions:
    torch.save(model, 'models/mod_temporal_torch' + '_' + str(i))
    # pickle.dump(model, open('models/mod_temporal_torch' + '_' + str(i), 'wb'))

# pickle.dump([mu, std], open('models/ustd_torch', 'wb'))
# pickle.dump([output_dimensions, state_dimensions, input_dimensions, history_size], open('models/dimensions', 'wb'))


# model = pickle.load(open('models/mod_temporal_torch', 'rb'))
# mu, std = pickle.load(open('models/ustd_torch', 'rb'))
# feat2 = pickle.load(open('../feat2', 'rb'))
# features = pickle.load(open('../features', 'rb'))
# past_command = pickle.load(open('../pastcomm', 'rb'))
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3, normalize=False)
# print('we')
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3)
# first = pickle.load(open('models/first_round', 'rb'))
# a = model.back_transform(first[:, :48], mu, std)
# print('here')