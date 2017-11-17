from pre_train.mlp_torch import MLP
import pickle
model = MLP(input_size = 48,output_size = 5, batch_size = 256, history_size = 4, cuda = True, epochs = 300)
model.init_datasets(fnames=['forza_2', 'forza_3'], fnames_test=['forza_4'])

loss, mu, std = model.train(500)
pickle.dump(model, open('models/mod_temporal_torch', 'wb'))
pickle.dump([mu, std], open('models/ustd_torch', 'wb'))
