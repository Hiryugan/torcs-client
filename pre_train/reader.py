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
    return smooth_dataset(npdata), np.array(tags)

import random
def construct_dataset(dataset, input_indices, state_indices, output_indices, history_size, all=False, is_train=False):
    id = len(input_indices)
    # id = input_indices.size(0)
    sd = len(state_indices)
    # sd = state_indices.size(0)
    print('a')
    od = len(output_indices)
    # od = output_indices.size(0)
    difficult = 0
    if all:
        train = np.zeros((dataset.shape[0] - history_size - 1, int(id*(history_size-1) + sd)))
    else:
        train = np.zeros((int((dataset.shape[0] / history_size)) ,int(id*(history_size-1) + sd) ))
    labels = np.zeros((train.shape[0], od))
    s = 0
    train2 = train.copy()
    # train2 = np.zeros((1, int(id*(history_size-1) + sd) ))
    labels2 = labels.copy()
    idx = 0
    # labels2 = np.zeros((1, od) )
    for i in range(train.shape[0]):
        rnd = random.randint(0, dataset.shape[0] - history_size - 1)
        if all:
            rnd = i
        # rnd = i
        countsteer = 0
        if np.abs(dataset[rnd + history_size-1, 3]) > 0.01 and is_train==True:
            countsteer += 1
            difficult += 1

        p = random.random()
        if p > 0.5 or countsteer > 0:
            # i = i - 1
            s += 1
            # labels2 = np.vstack((labels2, dataset[rnd, output_indices]))
            # train2 = np.vstack((train2, np.zeros((1, int(id*(history_size-1) + sd)))))
            # train2[-1, :sd] = dataset[rnd, state_indices]
            train2[idx, :sd] = dataset[rnd, state_indices]
            train2[idx, sd:] = dataset[rnd:rnd+history_size-1, input_indices].reshape(1, -1)
            labels2[idx, :] = dataset[rnd, output_indices]
            # train2[-1, sd:] = dataset[rnd:rnd+history_size-1, input_indices].reshape(1, -1)
            idx += 1
        labels[i, :] = dataset[rnd, output_indices]
        train[i, :sd] = dataset[rnd, state_indices]
        train[i, sd:] = dataset[rnd:rnd+history_size-1, input_indices].reshape(1, -1)

    # for i in range(train.shape[0]):
    #     if np.abs(train[i, 0]) < 0.1:
    #         s += 1
    print(train.shape, labels.shape, s, difficult)
    print(train2.shape, labels2.shape, s, difficult)
    if is_train:
        return train2[:idx], labels2[:idx]
    else:
        return train, labels

def smooth_dataset(dataset):
    # return dataset
    for j in range(3):
        for i in range(1, dataset.shape[0] -1):
            if dataset[i, 1] == 0 and dataset[i-1, 1] != 0 and dataset[i+1, 1] != 0:
                dataset[i, 1] = 0.5*(dataset[i-1, 1] + dataset[i+1, 1])
            # if dataset[i, 1] > 0.1:
            #     dataset[i, 1] = 1#*= 2
            # else:
            #     dataset[i, 1] = 0
            # if dataset[i - 1, 1] != 0 or dataset[i + 1, 1] != 0:
            #     dataset[i, 1] = 1
    # for j in range(2):
    #     for i in range(1, dataset.shape[0] -1):
    #         if dataset[i, 1] != 0 and dataset[i-1, 1] == 0 and dataset[i+1, 1] == 0:
    #             dataset[i, 1] = 0

    return dataset


def construct_dataset_lstm(dataset, input_indices, state_indices, output_indices, history_size, all=False, is_train=False):
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
        countsteer = 0
        # for j in range(0, history_size-1):
        #     if np.abs(dataset[rnd+j, 1]) < 1 and not all:
        #         countsteer += 1
        # if countsteer > 4:
        #     p = random.random()
        #
        #     if p < 0.5:
        #         i = i - 1
        #         s += 1
        labels[:, i, :] = dataset[rnd:rnd+history_size, output_indices]
        # print(i, rnd)
        train[:, i, :] = dataset[rnd:rnd+history_size, state_indices]#.reshape(1, -1)
    # print(train.shape, labels.shape)
    print(s)
    return train, labels
