import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# from  reader import get_data2, construct_dataset_lstm
import pickle
# from .pickler import load_model, save_model
import copy
from  pre_train.standard_nn import Standard_nn
torch.manual_seed(42)

class myLSTM(Standard_nn):
    def __init__(self, input_dimensions, output_dimensions, state_dimensions, batch_size, history_size, cuda, epochs):
        self.input_size = len(input_dimensions)
        self.output_size = len(output_dimensions)
        self.state_dimensions = len(state_dimensions)
        self.history_size = history_size
        # nn.LSTM.__init__(self, input_size=self.input_size, hidden_size=256, num_layers=1)
        Standard_nn.__init__(self, input_dimensions, output_dimensions, state_dimensions, batch_size, cuda, epochs)
        self.hidden_size = 256
        self.init_lstm()
        self.lr_decay = 0.8

    def init_lstm(self):
        self.lstmCell = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size)
        # self.lstmCell2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size)
        # self.lstmCell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size, nonlinearity='relu')
        # self.hn = Variable(torch.randn(self.history_size, self.batch_size, self.input_size))
        self.rnnCell = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, nonlinearity='relu')

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fce = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        # self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc5 = nn.Linear(self.hidden_size, self.output_size)
        self.hn = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        self.cn = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        self.output = Variable(torch.zeros(self.history_size, self.batch_size, self.output_size)).cuda()

    #@profile
    def forward2(self, x):
        # start = time.time()
        hn2 = self.hn.view(1, self.hn.size(0), -1).clone()
        s = time.time()
        # for i in range(x.size(0)):
        #
        #     if isinstance(self.lstmCell, nn.LSTMCell):
        #         self.hn, self.cn = self.lstmCell(x[i], (self.hn, self.cn))
        #     else:
        #         self.hn = self.lstmCell(x[i], self.cn)
        #     out = self.hn
        #     out = F.relu(self.fc1(out))
        #     out = self.fc2(out)
        #     self.output[i, :, :] = out
        # print('fir, ', time.time() - s)
        s = time.time()
        # out22, hn22, cn22 = self.lstmCell(x, )
        out2, hn2 = self.lstmCell2(x, hn2)
        # self.hn = hn2.view(hn2.size(1), hn2.size(2))
        out3 = F.relu(self.fc1(out2))
        out4 = self.fc2(out3)
        # print('sec,', time.time() - s)
        # print('\t', time.time() - start)
        return out4

    def forward(self, x, hn, cn=None):
        # hn2 = self.hn.view(1, self.hn.size(0), -1).clone()
        # if isinstance(self.lstmCell, nn.LSTMCell):
        #     hn2, cn2 = self.lstmCell(x, (hn2, self.cn))
        # else:
        #     out2, hn2 = self.lstmCell2(x, hn2)
        #
        # self.hn = hn2.view((hn2.size(1), hn2.size(2)))
        # x = F.selu(self.fce(x))
        if cn is None:
            out2, hn = self.rnnCell(x, hn)
        else:
            out2, (hn, cn) = self.lstmCell(x, (hn, cn))

        out3 = F.selu(self.fc1(out2))
        # out3, _ = self.lstmCell(out2)
        out4 = self.fc2(out3)
        # print('sec,', time.time() - s)
        # print('\t', time.time() - start)
        if cn is None:
            return out4, hn
        else:
            return out4, hn, cn

    def predict(self, x, hn, cn):

        self.itt = 0

        if hn is None:
            self.hn = Variable(torch.zeros(self.batch_size, self.hidden_size))
            self.cn = Variable(torch.zeros(self.batch_size, self.hidden_size))
        else:
            self.hn = hn
            self.cn = cn
        if hn is not None:
            x = x.view(1, x.size(0), x.size(1))
        self.output = Variable(torch.zeros(x.size(0), x.size(1), self.output_size))

        # for i in range(x.size(0)):
        if isinstance(self.lstmCell, nn.LSTMCell):
            self.hn, self.cn = self.lstmCell(x[i], (self.hn, self.cn))
        else:
            self.hn = self.lstmCell(x[i], self.hn)
        out = self.hn
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        if isinstance(self.lstmCell, nn.LSTMCell):
            return self.output, self.hn, self.cn
        else:
            return self.output, self.hn

    def cuda(self):
        super(myLSTM, self).cuda()
        self.lstmCell.cuda()
        # self.hn = self.hn.cuda()
        # self.cn = self.cn.cuda()
        # self.c0 = self.c0.cuda()

    def evaluate(self, datasets):
        """Evaluate a model on a data set."""
        # correct = 0.0
        s_tot = 0
        test_loss_tot = 0.0
        start = time.time()
        losses = []
        for dataset, labels in datasets:
            # for every
            # dataset, labels = datasett
            s = 0
            # self.hn = Variable(torch.zeros(self.batch_size, self.hidden_size))
            # self.cn = Variable(torch.zeros(self.batch_size, self.hidden_size))
            test_loss = 0.0
            for j in range(int(dataset.size(1) / self.batch_size)):
                i = random.randint(0, int(dataset.size(1) - self.batch_size) - 1)
                lookup_tensor = dataset[:, i:i + self.batch_size, :]
                target = labels[:, i:i + self.batch_size, :]

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

            losses.append(test_loss / s)
        print("evaluation test set: , time=%.2fs, \t train loss/sent=%.4f" %
              # ( test_loss.cpu().data.numpy() / s, time.time() - start))
              (time.time() - start, test_loss_tot / s_tot))
        print(losses)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    #@profile
    def train(self, epochs=None):
        # self.normalize(self.datasets, self.datasets_test)
        self.cuda()
        if epochs == None:
            epochs = self.epochs
        print(epochs)

        print(self.use_cuda)

        for ITER in range(1, epochs+1):
            s_tot = 0
            train_loss_tot = 0
            losses = []

            start = time.time()
            # h0 = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
            h0 = Variable(torch.cuda.FloatTensor(1, self.batch_size, self.hidden_size).fill_(0.0))
            # c0 = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
            c0 = Variable(torch.cuda.FloatTensor(1, self.batch_size, self.hidden_size).fill_(0.0))

            if ITER % 25 == 0 and ITER > 5:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay
            for dataset, labels in self.datasets:
                train_loss = 0.0
                s = 0
                # for every
                # self.hn = Variable(self.hn.data)
                # print(self.hn.size())
                # self.cn = Variable(self.cn.data)
                self.hn = Variable(h0.data.clone())
                self.cn = Variable(c0.data.clone())
                # dataset = dataset.view(self.history_size, -1, len(self.state_dimensions))
                # dataset, labels = datasett
                for j in range(int(dataset.size(1) / self.batch_size)):
                    # i = random.randint(0, int(dataset.size(1) - self.batch_size) - 1)
                    i = j*self.batch_size
                    self.hn = Variable(self.hn.data)
                    self.cn = Variable(self.cn.data)
                    # self.hn = Variable(torch.cuda.FloatTensor(self.batch_size, self.hidden_size).fill_(0.0))
                    # self.cn = Variable(torch.cuda.FloatTensor(self.batch_size, self.hidden_size).fill_(0.0))

                    # self.hn = Variable(h0.data.clone())
                    # self.cn = Variable(c0.data.clone())
                    lookup_tensor = dataset[:, i:i+self.batch_size, :]
                    target = labels[:, i:i+self.batch_size:, :]

                    scores, self.hn = self.forward(lookup_tensor, self.hn)
                    # scores, self.hn = self.forward(lookup_tensor, self.hn)
                    # scores2 = self.back_transform2(scores, self.mu, self.std)

                    # target2 = self.back_transform2(target, self.mu, self.std)
                    # torch.cuda.synchronize()
                    output = self.loss(scores, target)

                    train_loss += output.data[0]
                    train_loss_tot += output.data[0]
                    if ITER % 100 == 0:
                        print('we')
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
            # if ITER % epochs == 0:
            #     self.evaluate(self.datasets_test)
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
        self.lstmCell.cpu()
        self.rnnCell.cpu()
        self.mu = self.mu.cpu()
        self.std = self.std.cpu()
        return train_loss, self.mu, self.std
            # print("iter %r: test acc=%.4f" % (ITER, acc))

