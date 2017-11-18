from __future__ import print_function

from pre_train.reader import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
import random
import pickle

# kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
import numpy as np

import time
# @profile

def create_data(X):
    #remember that at the beginning, each row has the first 6 entries
    # as the outputs observed (aka desired for the mlp)

    data = np.zeros((X.shape[0]-1, X.shape[1]))
    labels = np.zeros((X.shape[0]-1, 5))

    data = X[1:, :]
    data[:, :5] = X[:-1, :5]
    labels = X[1:, :5]

    return data, labels

def preprocess(data):
    pass

def go():
    X, _ = get_data2(files='forza_3', folder='../data/')
    # u = X.mean(0)
    # std = X.std(0)
    # X = (X - u)
    # for i in range(X.shape[1]):
    #     if std[i] != 0:
    #         X[:, i] /= std[i]
    errs = []
    losses = []
    k = 500
    start = time.time()
    # np.random.permutation(X.shape[0])
    # X = np.random.permutation(X)
    # for i in range(10):

    train, target = create_data(X)

    mlp = MLPRegressor(verbose=0, random_state=0, alpha=0.01, hidden_layer_sizes=512, max_iter=1000, **{'solver': 'adam', 'learning_rate_init': 0.0001})
    mlp_gear = MLPClassifier(verbose=0, random_state=0, alpha=0.0001, hidden_layer_sizes=1024, max_iter=500, **{'solver': 'adam', 'learning_rate_init': 0.0001})
    train_gear = np.copy(train)
    train_gear = np.delete(train_gear, [0,1,3,4], axis=1)
    target_gear = np.copy(target)
    target_gear = np.delete(target_gear, [0,1,3,4], axis=1)
    target_gear = target_gear.squeeze()
    # res = mlp.fit(train, target)
    res_gear = mlp_gear.fit(train_gear, target_gear)
    # print(res)
    print(train.shape, target.shape)
    print( "Training set score: %f" % mlp_gear.score(train_gear, target_gear))
    print( "Training set loss: %f" % mlp_gear.loss_)
    # pred = mlp.predict(X[:k, 3:])
    # pred = mlp.predict(target)
    # mat = np.zeros((2,2))

    Xtest, _ = get_data2(files = 'forza_4', folder = '../data/')
   #  a = np.array([[  1.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00 ,  0.00000000e+00,
   #  0.00000000e+00 ,  0.00000000e+00 , -3.48266000e-04 , -9.82000000e-01,
   #  0.00000000e+00 ,  5.75910000e+03 ,  0.00000000e+00 ,  9.40000000e+01,
   #  0.00000000e+00 ,  1.00000000e+00 ,  9.42478000e+02 , -5.88597000e-03,
   # -2.78847000e-02 ,  1.79881000e-04 ,  0.00000000e+00 ,  0.00000000e+00,
   #  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   #  0.00000000e+00  , 0.00000000e+00  , 0.00000000e+00  , 0.00000000e+00,
   #  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   #  0.00000000e+00 ,  0.00000000e+00  , 0.00000000e+00 ,  0.00000000e+00,
   #  0.00000000e+00 , -3.33357000e-01 ,  0.00000000e+00 ,  0.00000000e+00,
   #  1.16041715e+02 , -1.15034455e+02 ,  3.45256000e-01 , -1.00000000e+00,
   # -1.00000000e+00 , -1.00000000e+00 , -1.00000000e+00 , -1.00000000e+00,
   #  0.00000000e+00]])
   #  Xtest = np.vstack((Xtest, a))
   #  Xtest = np.vstack((Xtest, a))
    xorig = Xtest.copy()
    # Xtest -= u
    # for i in range(X.shape[1]):
    #     if std[i] != 0:
    #         Xtest[:, i] /= std[i]
    test, target_test = create_data(Xtest)
    # train_gear = np.copy(train)
    test = np.delete(test, [0, 1, 3, 4], axis=1)
    # target_gear = np.copy(target)
    target_test = np.delete(target_test, [0, 1, 3, 4], axis=1)
    pred = mlp_gear.predict(test)
    print(mlp_gear.score(test, target_test))
    # for i in range(X.shape[1]):
    #     if std[i] != 0:
    #         test[ i] *= std[i]
    #         if i < 5:
    #             target_test[i] *= std[i]
    #             pred[i] *= std[i]
    # test += u
    # target_test += u[:5]
    # pred += u[:5]
    print(pred[:5])
    print(target_test[:5])
    print(xorig[1:, :5])
    #

    # print(pred[-5:])
    # print(target_test[-5:])
    # print(xorig[-5:, :5])
    #
    print(pred[2500:2505])
    print(target_test[2500:2505])
    print(xorig[2501:2506, :5])

    # errlist = [(a - b)**2 / k for (a, b) in zip(pred, target)]
    # for j in range(pred.shape[0]):
    #     mat[int(pred[j]), int(target[j])] += 1
    # errlist = [1 if a == b else 0 for (a, b) in zip(pred, target)]
    # for j in range(pred.shape[0]):
    #     v = 0 if pred[j] < 0.6 else 1
    #     mat[v, int(target[j])] += 1

    # print(list(sorted(errlist)))
    # err = sum(errlist)
    # print(err)
    # print(i, err / k)
    # losses.append(err / k)
    # errs.append(err)
    print(time.time() - start)
    plt.plot(mlp_gear.loss_curve_[1:], label='aaa')
    plt.show()

    # pickle.dump(res, open('models/mod_temporal2', 'wb') )
    # pickle.dump([u, std], open('models/ustd', 'wb'))

go()
# print(losses)
# print(errs)

#[array([ 0.1139626 ,  0.13472915,  0.0732063 ]), array([ 0.16472437,  0.16387508,  0.08740516]), array([ 0.12980534,  0.1593366 ,  0.06941525]), array([ 0.17634797,  0.19071809,  0.08039816]), array([ 0.13172097,  0.157337  ,  0.06169863]), array([ 0.13400031,  0.14290791,  0.06717543]), array([ 0.1090132 ,  0.12784908,  0.05887466]), array([ 0.10816669,  0.15821168,  0.07411749]), array([ 0.11288099,  0.14322833,  0.07627858]), array([ 0.16449471,  0.226323  ,  0.11529769])]

# print(__doc__)
# import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import datasets
#
# # different learning rate schedules and momentum parameters
# params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
#            'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
#            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
#            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
#            'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
#            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
#            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
#           {'solver': 'adam', 'learning_rate_init': 0.01}]
#
# labels = ["constant learning-rate", "constant with momentum",
#           "constant with Nesterov's momentum",
#           "inv-scaling learning-rate", "inv-scaling with momentum",
#           "inv-scaling with Nesterov's momentum", "adam"]
#
# plot_args = [{'c': 'red', 'linestyle': '-'},
#              {'c': 'green', 'linestyle': '-'},
#              {'c': 'blue', 'linestyle': '-'},
#              {'c': 'red', 'linestyle': '--'},
#              {'c': 'green', 'linestyle': '--'},
#              {'c': 'blue', 'linestyle': '--'},
#              {'c': 'black', 'linestyle': '-'}]
#
#
# def plot_on_dataset(X, y, ax, name):
#     # for each dataset, plot learning for each learning strategy
#     print("\nlearning on dataset %s" % name)
#     ax.set_title(name)
#     X = MinMaxScaler().fit_transform(X)
#     mlps = []
#     if name == "digits":
#         # digits is larger but converges fairly quickly
#         max_iter = 15
#     else:
#         max_iter = 400
#
#     for label, param in zip(labels, params):
#         print("training: %s" % label)
#         mlp = MLPClassifier(verbose=0, random_state=0,
#                             max_iter=max_iter, **param)
#         mlp.fit(X, y)
#         mlps.append(mlp)
#         print("Training set score: %f" % mlp.score(X, y))
#         print("Training set loss: %f" % mlp.loss_)
#     for mlp, label, args in zip(mlps, labels, plot_args):
#             ax.plot(mlp.loss_curve_, label=label, **args)
#
#
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# # load / generate some toy datasets
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# data_sets = [(iris.data, iris.target),
#              (digits.data, digits.target),
#              datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
#              datasets.make_moons(noise=0.3, random_state=0)]
#
# for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits',
#                                                     'circles', 'moons']):
#     plot_on_dataset(*data, ax=ax, name=name)
#
# fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()