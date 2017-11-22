import csv
import numpy as np


import csv
import numpy as np
import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_data(files=['alpine-1.csv'], folder='../data/'):

    #files = ['aalborg.csv', 'alpine-1.csv' , 'f-speedway.csv']
    # files = ['aalborg.csv']#, 'alpine-1.csv' , 'f-speedway.csv']

    data = []
    tags = []
    dim = 0
    for fname in files:
        f = open(folder + fname)
        reader = csv.reader(f, delimiter=',')

        i = 0
        for row in reader:
            row = [x for x in row if x != '']

            if i == 0:
                tags = row
                dim = len(row)
                print(dim)
            else:
                if len(row) != dim:
                    print('different number of columns at row {0}: had dim of {1} and got {2} dimensions'.format(i, dim, len(row)))
                else:
                    data.append([float(x) for x in row])
            i += 1

    # npdata = np.zeros((len(data), len(data[0])))
    # for i, row in enumerate(data):
    #     row = list(map(float, row))
    #     print(row)
    #     npdata[i] = np.array(row)
    npdata = np.array(data, dtype=np.float32)
    return npdata, np.array(tags)



def get_data2(files='alpine-1.csv', folder='../data/'):

    #files = ['aalborg.csv', 'alpine-1.csv' , 'f-speedway.csv']
    # files = ['aalborg.csv']#, 'alpine-1.csv' , 'f-speedway.csv']

    data = []
    tags = []
    dim = 0


    f = open(folder + files)

    for row in f:
        datarow = []
        vars = row.split(')')
        j = 0
        for var in vars:
            a = var.split(' ')
            # print(a)
            name = a[0]
            if name == '\n':
                break
            if len(a) > 2 and name != '(opponents':
                for v in a[1:]:
                    datarow.append(v)
            elif name != '(gear2' and name != '(opponents' and name != '(angle2':
                datarow.append(float(a[1]))
            elif name == '(angle2':
                raise 'angle2'
        data.append(datarow)

    # npdata = np.zeros((len(data), len(data[0])))
    # for i, row in enumerate(data):
    #     row = list(map(float, row))
    #     print(row)
    #     npdata[i] = np.array(row)
    npdata = np.array(data, dtype=np.float32)

    print("letto shape", npdata.shape)
    return npdata, np.array(tags)

import random
def construct_dataset(dataset, input_indices, state_indices, output_indices, history_size, all=False):
    id = len(input_indices)
    # id = input_indices.size(0)
    sd = len(state_indices)
    # sd = state_indices.size(0)
    od = len(output_indices)
    # od = output_indices.size(0)
    if all:
        train = np.zeros((dataset.shape[0] - history_size - 1, int(id*(history_size-1) + sd)))
    else:
        train  = np.zeros((int((dataset.shape[0] / history_size)) ,int(id*(history_size-1) + sd) ))
    labels = np.zeros((train.shape[0], od))

    for i in range(train.shape[0]):
        rnd = random.randint(0, dataset.shape[0] - history_size - 1)
        if all:
            rnd = i
        labels[i, :] = dataset[rnd, output_indices]
        train[i, :sd] = dataset[rnd, state_indices]
        train[i, sd:] = dataset[rnd:rnd+history_size-1, input_indices].reshape(1, -1)
    print(train.shape, labels.shape)
    return train, labels
