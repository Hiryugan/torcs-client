from pre_train.mlp_torch_2 import MLP
import pickle

# indices of the input dimensions for the past frames
# notice, 48 uses all dimensions but the opponent sensors
input_dimensions = [i for i in range(48)]
# use the first 5 dimensions as outputs (accel, brake, gear, steer, clutch)
output_dimensions = [0, 1, 2, 3, 4]
# dimensions of the state input (all the car state but opponents)
state_dimensions = [i for i in range(5, 48)]

# batch size actually helps very much for both convergence and running time of the iterations
model = MLP(input_size = len(input_dimensions),output_size = len(output_dimensions), state_size=len(state_dimensions), batch_size = 256, history_size = 4, cuda = True, epochs = 800)
model.init_datasets(fnames=['forza_2', 'forza_3'], fnames_test=['forza_4'])
#
loss, mu, std = model.train()
pickle.dump(model, open('models/mod_temporal_torch', 'wb'))
pickle.dump([mu, std], open('models/ustd_torch', 'wb'))
#
# model = pickle.load(open('models/mod_temporal_torch', 'rb'))
# mu, std = pickle.load(open('models/ustd_torch', 'rb'))
#
# first = pickle.load(open('models/first_round', 'rb'))
# a = model.back_transform(first[:, :48], mu, std)
# print('here')