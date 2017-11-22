from pre_train.mlp_torch_2 import MLP
import pickle

# indices of the input dimensions for the past frames
# notice, 48 uses all dimensions but the opponent sensors
input_dimensions = [i for i in range(5, 48)]
# use the first 5 dimensions as outputs (accel, brake, gear, steer, clutch)
# output_dimensions = [0, 1, 2, 3, 4]
output_dimensions = [3]#[0, 1, 2, 3, 4]
# dimensions of the state input (all the car state but opponents)
state_dimensions = [i for i in range(5, 48)]

dataset_names = ['forza', 'aalborg', 'alpine2', 'alpine', 'ruudskogen','cg_track_2']
dataset_names_test = ['cg_speedway_number1' , 'olethros_road_1',  'wheel2']
dataset_names2 = [x + '_2' for x in dataset_names]
dataset_names3 = [x + '_3' for x in dataset_names_test]
# batch size actually helps very much for both convergence and running time of the iterations
# model = MLP(input_dimensions = input_dimensions,output_dimensions = output_dimensions, state_dimensions=state_dimensions, batch_size = 256, history_size = 1, cuda = True, epochs = 35)
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3)

# loss, mu, std = model.train()
# pickle.dump(model, open('models/mod_temporal_torch', 'wb'))
# pickle.dump([mu, std], open('models/ustd_torch', 'wb'))
#
model = pickle.load(open('models/mod_temporal_torch', 'rb'))
mu, std = pickle.load(open('models/ustd_torch', 'rb'))
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3, normalize=False)
print('we')
# model.init_datasets(fnames=dataset_names2, fnames_test=dataset_names3)
# first = pickle.load(open('models/first_round', 'rb'))
# a = model.back_transform(first[:, :48], mu, std)
# print('here')