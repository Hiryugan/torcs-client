from pre_train.mlp_torch_2 import MLP
import pickle

input_size = [i for i in range(48)]
output_size = [0, 1, 2, 3, 4]
state_size = [i for i in range(5, 48)]
model = MLP(input_size = len(input_size),output_size = len(output_size), state_size=len(state_size), batch_size = 256, history_size = 4, cuda = True, epochs = 800)
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