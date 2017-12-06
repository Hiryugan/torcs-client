import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pre_train.reader import get_data2, construct_dataset
import pickle
from sklearn.decomposition import PCA

# from .pickler import load_model, save_model
import copy
torch.manual_seed(42)

class Standard_nn(nn.Module):
    def __init__(self, input_dimensions, output_dimensions, state_dimensions, batch_size, cuda, epochs):
        super(Standard_nn, self).__init__()
        self.output_size = len(output_dimensions)
        self.output_dimensions = output_dimensions
        self.state_size = len(state_dimensions)
        self.state_dimensions = state_dimensions
        self.input_size = len(input_dimensions)
        self.input_dimensions = input_dimensions

        self.batch_size = batch_size
        self.use_cuda = cuda
        self.epochs = epochs


        self.datasets = []
        self.datasets_orig = []
        self.datasets_test = []
        self.datasets_test_orig = []

        self.mu = Variable(torch.zeros(self.input_size))
        self.std = Variable(torch.zeros(self.input_size))

    def evaluate(self, datasets):
        """Evaluate a model on a data set."""
        # correct = 0.0
        s = 0
        it = 0
        test_loss = 0.0
        start = time.time()
        avg_prediction = Variable(torch.zeros(self.output_size))
        for dataset, labels in datasets:
            # for every
            # dataset, labels = datasett
            for i in range(int(dataset.size(0) / self.batch_size)):
                lookup_tensor = dataset[i:i + self.batch_size]
                target = labels[i:i + self.batch_size]

                scores = self.forward(lookup_tensor)
                # scores2 = self.back_transform2(scores, self.mu, self.std)

                # target2 = self.back_transform2(target, self.mu, self.std)
                output = self.loss(scores, target)
                test_loss += output.data[0]

                # backward pass
                self.zero_grad()
                output.backward()

                # update weights
                self.optimizer.step()
                s += 1

        print("evaluation test set: , time=%.2fs, \t train loss/sent=%.4f" %
              # ( test_loss.cpu().data.numpy() / s, time.time() - start))
              (time.time() - start, test_loss / s))


        return test_loss / s



    def init_datasets(self, fnames, fnames_test, construct_dataset_function, normalize=True):
        # fnames = ['forza_1']
        # fnames_test = ['forza_test']
        self.datasets = []
        self.datasets_test = []
        # means = torch.zeros(len(fnames), self.input_size)
        # vars = torch.zeros(len(fnames), self.input_size)


        for name in fnames:
            # try:
            #     f = open('../data/' + name + '.pickle', 'rb')
            #     data = pickle.load(f)
            #     print('caricato ' + name)
            # except:
            # f = open('../data/' + name + '.pickle', 'wb')
            data, _ = get_data2(name)
            # pickle.dump(data, f)
            print('scritto ' + name)
            data = Variable(torch.from_numpy(data))
            self.datasets.append(data)

        for name in fnames_test:
            # try:
            #     f = open('../data/' + name + '.pickle', 'rb')
            #     data = pickle.load(f)
            #     print('caricato test' + name)
        # except:
            # f = open('../data/' + name + '.pickle', 'wb')
            data, _ = get_data2(name)
            # pickle.dump(data, f)
            print('scritto test ' + name)
            data = Variable(torch.from_numpy(data))
            self.datasets_test.append(data)

        self.datasets_orig = copy.deepcopy(self.datasets)
        self.datasets_orig_test = copy.deepcopy(self.datasets_test)
        if normalize:
            self.mu, self.std, N = self.get_normalization(self.datasets)
            # print(self.mu, self.std)
        self.normalize(self.datasets, self.datasets_test, self.mu, self.std)
        self.mu[:5] = 0
        self.std[:5] = 1

        l = []
        l2 = []
        for data in self.datasets:
            # data2, label2 = construct_dataset(data.data.numpy(), [i for i in range(48)], [i for i in range(5, 48)], [0,1,2,3,4], 4)
            data2, label2 = construct_dataset_function(data.data.numpy(), self.input_dimensions, self.state_dimensions, self.output_dimensions, self.history_size, all=True, is_train=False)
            data2 = Variable(torch.FloatTensor(data2))
            label2 = Variable(torch.FloatTensor(label2))
            l.append((data2, label2))

        for data in self.datasets_test:
            # data2, label2 = construct_dataset(data.data.numpy(), [i for i in range(48)], [i for i in range(5, 48)], [0, 1,2,3,4], 4, all=True)
            data2, label2 = construct_dataset_function(data.data.numpy(), self.input_dimensions, self.state_dimensions, self.output_dimensions, self.history_size, all=True, is_train=False)
            data2 = Variable(torch.FloatTensor(data2))
            label2 = Variable(torch.FloatTensor(label2))
            l2.append((data2, label2))
        self.datasets = l
        self.datasets_test = l2

    # def get_normalization_pca(self, datasets):
    #     data = datasets[0]
    #     for i in range(1, len(datasets)):
    #         data = np.vstack((data, datasets[i]))
    #     self.pca = PCA(n_components=data.shape[1], svd_solver='full', whiten=True)
    #     res = self.pca.fit(data)

    def get_normalization(self, datasets):
        N = sum([len(x) for x in datasets])
        mu = Variable(torch.zeros(datasets[0].data.size(1)))
        std = Variable(torch.zeros(datasets[0].data.size(1)))
        var = Variable(torch.zeros(datasets[0].data.size(1)))
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
        # return Y
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
        # return Y
        X = Variable(Y.data.clone())
        for i in range(X.size(1)):
            if std.data[i] > 0:
                for j in range(X.size(0)):
                    X[j, i] = X[j, i] * std[i]
        X += mu[:X.size(1)]
        return X

    def transform2(self, Y, mu, std):
        # return Y
        X = Y
        X = X - mu[:X.size(1)]
        for i in range(X.size(1)):
            if std.data[i] > 0:
                X[:, i] /= std[i]
        return X

    def back_transform2(self, Y, mu, std):
        # return Y
        X = Y

        for i in range(X.size(1)):
            if std.data[i] > 0:
                stdv = Variable(torch.ones(1, X.size()[0]))
                if self.use_cuda:
                    stdv = stdv.cuda()

                X[:, i] = X[:, i].clone()* stdv
        X += mu[:X.size(1)]
        return X

    def normalize(self, datasets, datasets_test, mu, std):

        for i in range(len(datasets)):
            datasets[i][:, 5:] = self.transform(datasets[i][:, 5:], mu[5:], std[5:])

        for i in range(len(datasets_test)):
            datasets_test[i][:, 5:] = self.transform(datasets_test[i][:, 5:], mu[5:], std[5:])
        return mu, std

    def cuda(self):
        super(Standard_nn, self).cuda()
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
        for ITER in range(1, epochs+1):
            train_loss = 0.0
            start = time.time()
            s = 0
            for dataset, labels in self.datasets:
                # for every
                # dataset, labels = datasett
                for i in range(int(dataset.size(0) / self.batch_size)):
                    lookup_tensor = dataset[i:i+self.batch_size]
                    target = labels[i:i+self.batch_size]

                    scores = self.forward(lookup_tensor)
                    # scores2 = self.back_transform2(scores, self.mu, self.std)
                    #
                    # target2 = self.back_transform2(target, self.mu, self.std)
                    output = self.loss(scores, target)
                    train_loss += output.data[0]
                    if ITER % 50 == 0:
                        print('we')
                    # backward pass

                    self.zero_grad()
                    output.backward()

                    # update weights
                    self.optimizer.step()
                    s += 1

            print("iter %r: train loss/sent=%.4f, time=%.2fs" %
                  (ITER, train_loss / s, time.time() - start))
            if ITER % 5 == 0:
                self.evaluate(self.datasets_test)
            # evaluate
        # self.save_model(self, open('models/mod_temporal_torch', 'wb'))
        # pickle.dump(self, open('models/mod_temporal_torch', 'wb') )
        # self.save_model([self.mu, self.std], open('models/ustd_torch', 'wb'))
        self.use_cuda = False
        self.datasets = None
        self.datasets_test = None
        self.datasets_orig = None
        self.datasets_orig_test = None

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
