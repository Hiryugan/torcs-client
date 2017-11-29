from pre_train.mlp_torch_2 import MLP
from pre_train.mlp import MLP2
from pre_train.reader import construct_dataset_lstm, construct_dataset
from pre_train.lstm import myLSTM
import pickle

# indices of the input dimensions for the past frames
# notice, 48 uses all dimensions but the opponent sensors
# use the first 5 dimensions as outputs (accel, brake, gear, steer, clutch)
# output_dimensions = [0, 1, 2, 3, 4]
output_dimensions = [0, 1, 3]#[0, 1, 2, 3, 4]
# dimensions of the state input (all the car state but opponents)
# state_dimensions = [i for i in range(5, 48)]
state_dimensions = [6, 15, 16, 17] + list(range(18, 38)) + [42]
# input_dimensions = [3]+[i for i in range(5, 48)]
input_dimensions = state_dimensions
history_size = 5
batch_size = 256
cuda = True
epochs = 500
# todo: correct data with -1, avoid frames without change in angle

dataset_names = [ 'aalborg', 'alpine', 'alpine2', 'brondehach', 'etrack2', 'cg_speedway_number1']
dataset_names_test = ['wheel1', 'aalborg' , 'alpine', 'alpine2', 'spring', 'wheel2', 'cg_speedway_number1', 'olethros_road_1', 'brondehach', 'etrack3']
dataset_names2 = [x + '_2' for x in dataset_names]
dataset_names3 = [x + '_3' for x in dataset_names_test]

construct_fun = construct_dataset_lstm
model = myLSTM(input_dimensions = input_dimensions,output_dimensions = output_dimensions, state_dimensions=state_dimensions, batch_size = batch_size, history_size = history_size, cuda = cuda, epochs = epochs)

# construct_fun = construct_dataset
# model = MLP2(input_dimensions = input_dimensions,output_dimensions = output_dimensions, state_dimensions=state_dimensions, batch_size = batch_size, history_size = history_size, cuda = cuda, epochs = epochs)
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3, normalize=True)
model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3, construct_dataset_function=construct_fun, normalize=True)

loss, mu, std = model.train()
for i in output_dimensions:
    pickle.dump(model, open('models/mod_temporal_torch' + '_' + str(i), 'wb'))

pickle.dump([mu, std], open('models/ustd_torch', 'wb'))
pickle.dump([output_dimensions, state_dimensions, input_dimensions, history_size], open('models/dimensions', 'wb'))


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