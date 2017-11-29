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

def get_data2(files='alpine-1.csv', folder='../data/'):

    #files = ['aalborg.csv', 'alpine-1.csv' , 'f-speedway.csv']
    # files = ['aalborg.csv']#, 'alpine-1.csv' , 'f-speedway.csv']

    data = []
    tags = []
    dim = 0


    f = open(folder + files)

    s = 0
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

        if datarow[18] != -1:
            # s += 1
            data.append(datarow)
        else:
            s+=1
    print(s)
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
        train = np.zeros((int((dataset.shape[0] / history_size)) ,int(id*(history_size-1) + sd) ))
    labels = np.zeros((train.shape[0], od))
    s = 0
    for i in range(train.shape[0]):
        rnd = random.randint(0, dataset.shape[0] - history_size - 1)
        if all:
            rnd = i
        # rnd = i

        if np.abs(dataset[rnd, 3]) < 0.5 and not all:
            p = random.random()

            if p < -1:
                i = i - 1
                s += 1
        labels[i, :] = dataset[rnd, output_indices]
        train[i, :sd] = dataset[rnd, state_indices]
        train[i, sd:] = dataset[rnd:rnd+history_size-1, input_indices].reshape(1, -1)

    # for i in range(train.shape[0]):
    #     if np.abs(train[i, 0]) < 0.1:
    #         s += 1
    print(train.shape, labels.shape, s)
    return train, labels


def construct_dataset_lstm(dataset, input_indices, state_indices, output_indices, history_size, all=False):
    id = len(input_indices)
    # id = input_indices.size(0)
    sd = len(state_indices)
    # sd = state_indices.size(0)
    od = len(output_indices)
    # od = output_indices.size(0)
    nbatches = int(dataset.shape[0] / history_size)
    if all:
        train = np.zeros((history_size, nbatches, sd))
    else:
        train  = np.zeros((history_size , nbatches, sd) )
    labels = np.zeros((history_size, train.shape[1], od))
    s = 0
    for i in range(train.shape[1]):
        rnd = random.randint(0, dataset.shape[0] - history_size - 1)
        if all:
            rnd = i
        rnd = i*history_size
        if np.abs(dataset[rnd, 3]) < 0.5 and not all:
            p = random.random()

            if p < -1:
                i = i - 1
                s += 1
        labels[:, i, :] = dataset[rnd:rnd+history_size, output_indices]
        # print(i, rnd)
        train[:, i, :] = dataset[rnd:rnd+history_size, state_indices]#.reshape(1, -1)
    # print(train.shape, labels.shape)
    return train, labels
