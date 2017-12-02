import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pre_train.reader import get_data2
import pickle
# from .pickler import load_model, save_model
import copy
torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, batch_size, history_size, cuda, epochs):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.history_size = history_size
        self.use_cuda = cuda
        self.epochs = epochs

        self.fc1 = nn.Linear(input_size*history_size, 1024)
        self.drop = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.datasets = []
        self.datasets_orig = []
        self.datasets_test = []
        self.datasets_test_orig = []
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        x = self.fc3(x)
        # x[0] = F.sigmoid(x[0])
        # x[1] = F.sigmoid(x[1])
        # x[3] = F.tanh(x[3])
        # x[4] = F.sigmoid(x[4])
        return x



    def evaluate(self, datasets):
        """Evaluate a model on a data set."""
        # correct = 0.0
        s = 0
        it = 0
        test_loss = 0.0
        start = time.time()
        avg_prediction = Variable(torch.zeros(self.output_size))
        for dataset in datasets:
            for i in range(dataset.size(0) - self.history_size):
                lookup_tensor = dataset[i:i+self.history_size].view(1, -1)
                scores = self.forward(lookup_tensor)
                scores2 = self.back_transform(scores, self.mu, self.std)
                target = dataset[i+self.history_size][:self.output_size].view(1, -1)
                target2 = self.back_transform(target, self.mu, self.std)
                output = self.loss(scores2, target2)
                # torch.sum((target2 - scores2)**2)
                # test_loss += torch.sum((target2 - scores2)**2)
                if it % 5 == 0:
                    it += 0
                    # print('ciao')
                it += 1
                test_loss += output.data[0]
                # backward pass
                self.zero_grad()
                # output.backward()
                # update weights
                # optimizer.step()
                s += 1

        print("evaluation test set: train loss/sent=%.4f, time=%.2fs" %
              # ( test_loss.cpu().data.numpy() / s, time.time() - start))
              ( test_loss / s, time.time() - start))


        return test_loss / s



    def init_datasets(self, fnames, fnames_test):
        fnames = ['forza_1']
        fnames_test = ['forza_test']
        self.datasets = []
        self.datasets_test = []
        means = torch.zeros(len(fnames), self.input_size)
        vars = torch.zeros(len(fnames), self.input_size)


        for name in fnames:
            data, _ = get_data2(name)
            data = Variable(torch.from_numpy(data))
            self.datasets.append(data)

        for name in fnames_test:
            data, _ = get_data2(name)
            data = Variable(torch.from_numpy(data))
            self.datasets_test.append(data)
        self.datasets_orig = copy.deepcopy(self.datasets)
        self.datasets_orig_test = copy.deepcopy(self.datasets_test)
        self.mu, self.std = self.normalize(self.datasets, self.datasets_test)

    def get_normalization(self, datasets):
        N = sum([len(x) for x in datasets])
        mu = Variable(torch.zeros(self.input_size))
        std = Variable(torch.zeros(self.input_size))
        var = Variable(torch.zeros(self.input_size))
        # if self.use_cuda:
        #     mu, std, var = mu.cuda(), std.cuda(), var.cuda()
        for data in datasets:
            mu += torch.sum(data, 0)
        mu /= N
        for data in datasets:
            var += torch.var(data, 0) * len(data)
            std += sum((data - mu)**2, 0)
        std /= N
        std = std.sqrt()
        var /= N
        var = var.sqrt()
        return mu, var, N


    def transform(self, Y, mu, std):
        X = Variable(Y.data.clone())
        # print(X)
        # print(X.size(1))
        # print(mu[:X.size(1)])
        X = X - mu[:X.size(1)]
        for i in range(X.size(1)):
            if std.data[i] > 0:
                X[:, i] /= std[i]
        return X

    def back_transform(self, Y, mu, std):
        X = Variable(Y.data.clone())
        for i in range(X.size(1)):
            if std.data[i] > 0:
                for j in range(X.size(0)):
                    X[j, i] = X[j, i] * std[i]
        X += mu[:X.size(1)]
        return X

    def transform2(self, Y, mu, std):
        X = Y
        X = X - mu[:X.size(1)]
        for i in range(X.size(1)):
            if std.data[i] > 0:
                X[:, i] /= std[i]
        return X

    def back_transform2(self, Y, mu, std):
        X = Y

        for i in range(X.size(1)):
            if std.data[i] > 0:
                stdv = Variable(torch.ones(1, X.size()[0]))
                if self.use_cuda:
                    stdv = stdv.cuda()

                X[:, i] = X[:, i].clone()* stdv
        X += mu[:X.size(1)]
        return X

    def normalize(self, datasets, datasets_test):
        mu, std, N = self.get_normalization(datasets)
        for i in range(len(datasets)):
            datasets[i] = self.transform(datasets[i], mu, std)

        for i in range(len(datasets_test)):
            datasets_test[i] = self.transform(datasets_test[i], mu, std)

        if self.use_cuda:
            self.cuda()
            for i, x in enumerate(datasets):
                datasets[i] = x.cuda()
            for i, x in enumerate(datasets_test):
                datasets_test[i] = x.cuda()
            for i, x in enumerate(self.datasets_orig):
                self.datasets_orig[i] = x.cuda()
            for i, x in enumerate(self.datasets_test_orig):
                self.datasets_test_orig[i] = x.cuda()
            # self.datasets_orig = self.datasets_orig.cuda()
            # self.datasets_test_orig = self.datasets_test_orig.cuda()
            mu = mu.cuda()
            std = std.cuda()
        return mu, std

    # @profile
    def train(self, epochs=None):
        # self.normalize(self.datasets, self.datasets_test)

        if epochs == None:
            epochs = self.epochs

        for ITER in range(1, epochs+1):
            train_loss = 0.0
            start = time.time()
            s = 0
            for dataset in self.datasets:
                # for every
                for i in range(int(dataset.size(0)/ ((self.history_size+1)*self.batch_size))):
                    # r = random.randint(0, dataset.size(0)-5)
                    rlist = [random.randint(0, dataset.size(0)-self.history_size-1) for j in range(self.batch_size)]
                    # forward pass
                    # lookup_tensor = dataset[r:r+1].view(1, -1)
                    lookup_tensor = Variable(torch.zeros(self.batch_size, self.input_size * self.history_size))
                    if self.use_cuda:
                        lookup_tensor = lookup_tensor.cuda()


                    for j in range(self.batch_size):
                        lookup_tensor[j, :] = dataset[rlist[j]:rlist[j]+self.history_size].view(1, -1)

                    if ITER % 50 == 0:
                        print('here')
                    scores = self.forward(lookup_tensor)
                    scores2 = self.back_transform2(scores, self.mu, self.std)
                    # target = dataset[r+4]
                    target = Variable(torch.zeros(self.batch_size, self.output_size))
                    if self.use_cuda:
                        target = target.cuda()
                    for j in range(self.batch_size):
                        target[j, :] = dataset[rlist[j]+self.history_size][:self.output_size]

                    target2 = self.back_transform2(target, self.mu, self.std)
                    output = self.loss(scores2, target2)
                    train_loss += output.data[0]

                    # backward pass
                    self.zero_grad()
                    output.backward()

                    # update weights
                    self.optimizer.step()
                    s += 1

            print("iter %r: train loss/sent=%.4f, time=%.2fs" %
                  (ITER, train_loss / s, time.time() - start))
            if ITER % 100 == 0:
                self.evaluate(self.datasets_test)
            # evaluate
        # self.save_model(self, open('models/mod_temporal_torch', 'wb'))
        # pickle.dump(self, open('models/mod_temporal_torch', 'wb') )
        # self.save_model([self.mu, self.std], open('models/ustd_torch', 'wb'))
        self.use_cuda = False
        self.cpu()
        self.mu = self.mu.cpu()
        self.std = self.std.cpu()
        return train_loss, self.mu, self.std
            # print("iter %r: test acc=%.4f" % (ITER, acc))

    # def save_model(self, model, file):
    #     pickle.dump(model, file)
    #
    # def load_model(self, file):
    #     mod = pickle.load(file)
    #     return mod
    #
    #
