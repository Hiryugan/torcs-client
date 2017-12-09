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

model = pickle.load(open(file, 'rb'))
model = model.cpu()

pickle.dump(model, open(file + '_cpu', 'wb'))