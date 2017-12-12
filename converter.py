import pickle
import torch
import argparse


parser = argparse.ArgumentParser(
        description=''
    )
parser.add_argument(
    '-f',
    '--file',
    help='Model file path.'
)

args = parser.parse_args()
file = args.file

model = torch.load(file, lambda storage, location: storage)
# torch.save(model, file + '.')



w1 = model.fc1.weight.data.numpy()
b1 = model.fc1.bias.data.numpy()

w2 = model.fc2.weight.data.numpy()
b2 = model.fc2.bias.data.numpy()

w3 = model.fc3.weight.data.numpy()
b3 = model.fc3.bias.data.numpy()

def selu():
    pass

def forward_numpy(x):
    x = x @ w1.transpose() + b1
    x = selu(x)
    x = x @ w2.transpose() + b2
    x = selu(x)
    x @ w3.transpose() + b3
    return x

mu = model.mu.data.numpy()
std = model.std.data.numpy()
history_size = model.history_size
output_dimensions = model.output_dimensions
input_dimensions = model.input_dimensions
state_dimensions = model.state_dimensions

d = {}
d['mu'] = mu
d['std'] = std
d['state_dimensions'] = state_dimensions
d['output_dimensions'] = output_dimensions
d['input_dimensions'] = input_dimensions
d['history_size'] = model.history_size
d['w1'] = w1
d['b1'] = b1
d['w2'] = w2
d['b2'] = b2
d['w3'] = w3
d['b3'] = b3

pickle.dump(d, open(file + '_dict', 'wb'))

def transform(Q, mu, std):
    Q = Q - mu[:Q.shape[1]]
    Q /= std
    return Q


def back_transform(Q, mu, std):
    Q = Q * std
    Q += mu[:Q.shape[1]]
    return Q