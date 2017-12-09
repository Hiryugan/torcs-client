import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import shuffle
import numpy as np
from pre_train.reader import get_data2, construct_dataset_lstm
import pickle
# from .pickler import load_model, save_model
import copy
from pre_train.standard_nn import Standard_nn
torch.manual_seed(42)

class MLP2(Standard_nn):
    def __init__(self, input_dimensions, output_dimensions, state_dimensions, batch_size, history_size, cuda, epochs):
        self.input_size = len(input_dimensions)
        self.output_size = len(output_dimensions)
        self.state_dimensions = len(state_dimensions)
        self.history_size = history_size
        # nn.LSTM.__init__(self, input_size=self.input_size, hidden_size=256, num_layers=1)
        Standard_nn.__init__(self, input_dimensions, output_dimensions, state_dimensions, batch_size, cuda, epochs)

        self.lr_decay = 0.8
        self.fc1 = nn.Linear(self.input_size * (history_size - 1) + self.state_size, 128)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_size)
        # self.fc1.weight.data = nn.init.orthogonal(self.fc1.weight.data)
        # self.fc2.weight.data = nn.init.orthogonal(self.fc2.weight.data)
        # self.fc3.weight.data = nn.init.orthogonal(self.fc3.weight.data)
        self.loss = nn.MSELoss()
        self.lossbreak = nn.BCELoss()
        # self.conv1 = nn.Conv1d()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001 / 20)
        # self.optimizer = optim.SGD(self.parameters(), lr=0.05 / 20, momentum=0.5, weight_decay=0.00001, nesterov=True)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = self.drop(x)
        # x = torch.cat((x, xorig), 1)
        x = F.selu(self.fc2(x))
        # x = self.drop2(x)
        # x = torch.cat((x, xorig), 1)
        x = self.fc3(x)
        # x[0] = F.sigmoid(x[0])
        # x[1] = F.sigmoid(x[1])
        # x[3] = F.tanh(x[3])
        # x = F.sigmoid(x)
        # x = F.tanh(x)
        return x



    def evaluate(self, datasets):
        """Evaluate a model on a data set."""
        # correct = 0.0
        s_tot = 0
        it = 0
        test_loss_tot = 0.0
        start = time.time()
        avg_prediction = Variable(torch.zeros(self.output_size))
        losses = []
        for dataset, labels in datasets:
            # for every
            s = 0
            test_loss = 0.0
            # dataset, labels = datasett
            for j in range(int(dataset.size(0) / (self.batch_size))):
                i = j*self.batch_size
                lookup_tensor = dataset[i:i + self.batch_size]
                target = labels[i:i + self.batch_size]

                scores = self.forward(lookup_tensor)
                # scores2 = self.back_transform2(scores, self.mu, self.std)

                # target2 = self.back_transform2(target, self.mu, self.std)
                output = self.loss(scores, target)
                test_loss += output.data[0]
                test_loss_tot += output.data[0]

                # backward pass
                self.zero_grad()
                # output.backward()

                # update weights
                # self.optimizer.step()
                s += 1
                s_tot += 1
            losses.append(test_loss/ s)
        print("evaluation test set: , time=%.2fs, \t train loss/sent=%.4f" %
              # ( test_loss.cpu().data.numpy() / s, time.time() - start))
              (time.time() - start, test_loss_tot / s_tot))
        print(losses)


        return test_loss / s


    def cuda(self):
        super(MLP2, self).cuda()
        if self.use_cuda:
            for i, (x, y) in enumerate(self.datasets):
                self.datasets[i] = (x.cuda(), y.cuda())
            for i, (x, y) in enumerate(self.datasets_test):
                self.datasets_test[i] = (x.cuda(), y.cuda())
            # for i, (x, y) in enumerate(self.datasets_orig):
            #     self.datasets_orig[i] = (x.cuda(), y.cuda())
            # for i, (x, y) in enumerate(self.datasets_test_orig):
            #     self.datasets_test_orig[i] = (x.cuda(), y.cuda())
            # self.datasets_orig = self.datasets_orig.cuda()
            # self.datasets_test_orig = self.datasets_test_orig.cuda()
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()

    # @profile

    def train(self, epochs=None):
        # self.normalize(self.datasets, self.datasets_test)
        self.cuda()
        if epochs == None:
            epochs = self.epochs
        print(epochs)
        print(self.use_cuda)
        mydataset = self.datasets[0][0]
        mylabels = self.datasets[0][1]

        for i in range(1, len(self.datasets)):
            mydataset = torch.cat((mydataset, self.datasets[i][0]))
            mylabels = torch.cat((mylabels, self.datasets[i][1]))

        while len(self.datasets) > 0:
            del self.datasets[0]

        while len(self.datasets_test) > 0:
            del self.datasets_test[0]

        for ITER in range(1, epochs + 1):
            losses = []
            start = time.time()
            s_tot = 0.0
            train_loss_tot = 0.0

            perm = torch.randperm(mydataset.size(0)).cuda()
            mydataset = mydataset[perm]
            mylabels = mylabels[perm]

            for j in range(int(mydataset.size(0) / self.batch_size)):

                train_loss = 0.0
                s = 0.0


                i = random.randint(0, mydataset.size(0) - self.batch_size - 1)

                lookup_tensor = mydataset[i:i+self.batch_size]
                target = mylabels[i:i+self.batch_size]

                scores = self.forward(lookup_tensor)
                # scores2 = self.back_transform2(scores, self.mu, self.std)

                # target2 = self.back_transform2(target, self.mu, self.std)
                output = self.loss(scores, target)
                train_loss += output.data[0]
                train_loss_tot += output.data[0]
                # if ITER % 10 == 0:
                #     print('we')
                # backward pass

                self.zero_grad()
                output.backward()

                # update weights
                self.optimizer.step()
                s += 1
                s_tot += 1

            print("iter %r: train loss/sent=%.4f, time=%.2fs" %
                  (ITER, train_loss_tot / s_tot, time.time() - start))
            # if ITER % 5 == 0:
            #     self.evaluate(self.datasets_test)
            # evaluate
            if ITER % 20 == 0 and ITER > 30:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay
            if ITER < 5:
                print('we')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 1.82056
        # self.save_model(self, open('models/mod_temporal_torch', 'wb'))
        # pickle.dump(self, open('models/mod_temporal_torch', 'wb') )
        # self.save_model([self.mu, self.std], open('models/ustd_torch', 'wb'))
        self.use_cuda = False
        self.datasets = None
        self.datasets_test = None
        self.datasets_orig = None
        self.datasets_orig_test = None
        #
        self.cpu()
        self.mu = self.mu.cpu()
        self.std = self.std.cpu()
        return train_loss, self.mu, self.std
        # print("iter %r: test acc=%.4f" % (ITER, acc))

    def train2(self, epochs=None):
        # self.normalize(self.datasets, self.datasets_test)
        self.cuda()
        if epochs == None:
            epochs = self.epochs
        print(epochs)
        print(self.use_cuda)

        for ITER in range(1, epochs+1):
            losses = []
            start = time.time()
            s_tot = 0.0
            train_loss_tot = 0.0

            shuffle(self.datasets)
            for dataset, labels in self.datasets:
                train_loss = 0.0
                s = 0.0
                # for every
                # dataset, labels = datasett
                # for j in range(int(dataset.size(0) / (self.batch_size*self.history_size))):
                # d1 = self.datasets[0]
                # for d in datasets[1:]:

                for j in range(int(dataset.size(0) / (self.batch_size))):
                # for j in range(50):
                    x = random.randint(0, len(self.datasets) - 1)
                    i = random.randint(0, self.datasets[x][0].size(0) - self.batch_size - 1)
                    y = random.randint(0, len(self.datasets) - 1)
                    k = random.randint(0, self.datasets[y][0].size(0) - self.batch_size - 1)
                    # i = random.randint(0, dataset.size(0) - self.batch_size - 1)
                    bs2 = int(self.batch_size / 2)
                    # i = j*self.batch_size
                    lookup_tensor = self.datasets[x][0][i:i+int(self.batch_size/2)]
                    # target = labels[i:i+self.batch_size]
                    target = self.datasets[x][1][i:i+int(self.batch_size/2)]

                    lookup_tensor2 = self.datasets[y][0][k:k + int(self.batch_size / 2)]
                    # target = labels[i:i+self.batch_size]
                    target2 = self.datasets[y][1][k:k + int(self.batch_size /2)]
                    lookup_tensor = torch.cat((lookup_tensor, lookup_tensor2))
                    target = torch.cat((target, target2))
                    scores = self.forward(lookup_tensor)
                    # scores2 = self.back_transform2(scores, self.mu, self.std)

                    # target2 = self.back_transform2(target, self.mu, self.std)
                    output = self.loss(scores, target)
                    train_loss += output.data[0]
                    train_loss_tot += output.data[0]
                    # if ITER % 10 == 0:
                    #     print('we')
                    # backward pass

                    self.zero_grad()
                    output.backward()

                    # update weights
                    self.optimizer.step()
                    s += 1
                    s_tot += 1
                losses.append(train_loss / s)
            print(losses)
            print("iter %r: train loss/sent=%.4f, time=%.2fs" %
                  (ITER, train_loss_tot / s_tot, time.time() - start))
            # if ITER % 5 == 0:
            #     self.evaluate(self.datasets_test)
            # evaluate
            if ITER % 25 == 0 and ITER > 30:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay
            if ITER < 5:
                print('we')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 1.82056
        # self.save_model(self, open('models/mod_temporal_torch', 'wb'))
        # pickle.dump(self, open('models/mod_temporal_torch', 'wb') )
        # self.save_model([self.mu, self.std], open('models/ustd_torch', 'wb'))
        self.use_cuda = False
        self.datasets = None
        self.datasets_test = None
        self.datasets_orig = None
        self.datasets_orig_test = None
        #
        # self = self.cpu()
        # self.mu = self.mu.cpu()
        # self.std = self.std.cpu()
        return train_loss, self.mu, self.std
            # print("iter %r: test acc=%.4f" % (ITER, acc))
