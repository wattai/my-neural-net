# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:51:21 2018

@author: wattai
"""

# If you want to use GPU, you can change from "numpy" to "cupy" below import.
import cupy as xp

from chainer import cuda
# from scipy.signal import convolve2d
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import sys

import optimizers
import layers as L
import utilities as ut
import initializers as init
import model


def to_onehot(y, n_class=None):
    xp = cuda.get_array_module(y)
    if n_class is None:
        encoder = preprocessing.OneHotEncoder()
        return encoder.fit_transform(y[:, None]).toarray()
    else:
        return xp.eye(n_class)[y]


class Mynet(model.Model):
    def __init__(self, lossfunc, optimizer, batch_size):
        super().__init__(lossfunc, optimizer, batch_size)

        self.conv0 = L.Convolution_(n_filter=8, filter_size=(3, 3), stride=1)
        self.conv1 = L.Convolution_(n_filter=16, filter_size=(3, 3), stride=1)
    
        self.fc0 = L.Linear_(output_size=1024)
        self.fc1 = L.Linear_(output_size=10)
    
        self.bn0 = L.BatchNormalization_()
        self.bn1 = L.BatchNormalization_()
        self.bn4 = L.BatchNormalization_()
    
        self.acti0 = L.ELU()
        self.acti1 = L.ELU()
        self.acti4 = L.ELU()
    
        self.pool0 = L.MaxPooling(7, 7)
        self.pool1 = L.MaxPooling(5, 5)
    
        self.flat = L.Flatten()
    
        self.drop0 = L.Dropout(0.5)
        self.drop1 = L.Dropout(0.5)
        
        self.layers = [self.conv0,
                       self.acti0,
                       self.pool0,
                       self.bn0,
                       #self.drop0, 
                       
                       self.conv1,
                       self.acti1,
                       self.pool1,
                       self.bn1,
                       #self.drop1,
                       
                       self.flat,
                       
                       self.fc0,
                       self.acti4,
                       self.bn4,
                       
                       self.fc1,
                       ]


if __name__ == '__main__':


    
    
    iter_num = 100000

    # X, t = digits = datasets.load_digits(n_class=10, return_X_y=True)
    # X = X.astype(np.float32) / 255.
    # t = t.astype(np.int32)
    mnist = datasets.fetch_mldata('MNIST original')
    X, t = mnist.data.astype(xp.float32) / 255., mnist.target.astype(xp.int32)
    # X, t = np.asarray(X), np.asarray(t)

    minisize = 70000 // 10  # //10 is reducing the number of sample.
    mini = xp.random.choice(X.shape[0], minisize).tolist()
    X, t = X[mini], t[mini]

    T = to_onehot(t, n_class=10).astype(xp.float32)
    X_train, X_test, T_train, T_test = train_test_split(X, T,
                                                        test_size=1/7,
                                                        shuffle=True)
    X_train, X_test = xp.array(X_train), xp.array(X_test)
    T_train, T_test = xp.array(T_train), xp.array(T_test)

    train_size = X_train.shape[0]
    batch_size = 100  # X_test.shape[0]

    # reshape to fit the size of array for Convolutinal Layer. -----------
    width = xp.sqrt(X.shape[1]).astype(xp.int32).tolist()
    X_train = X_train.reshape(X_train.shape[0], 1, width, width
                              ).astype(xp.float32)
    X_test = X_test.reshape(X_test.shape[0], 1, width, width
                            ).astype(xp.float32)
    # --------------------------------------------------------------------
    epoch_size = train_size // batch_size

    print('begin training...')
    network = Mynet(
            lossfunc=L.Softmax_cross_entropy_with_variational_regularizer(
                    variational=False),  # True にすると正則化係数もパラメータになる．
            optimizer=optimizers.Adam(l1=1e-4, l2=1e-4),
            batch_size=batch_size,
            )

    loss_history = []
    acc_history = []
    val_loss_history = []
    val_acc_history = []

    start = time.time()

    for index in range(iter_num):
        batch_choice = xp.random.choice(train_size, batch_size).tolist()

        x_batch = xp.asarray(X_train[batch_choice])
        t_batch = xp.asarray(T_train[batch_choice])

        network.update(x_batch, t_batch)

        if (index % epoch_size == 0):

            Y_train = network.forward(X_train)

            train_loss = network.compute_loss(Y_train, T_train)
            loss_history.append(train_loss)

            train_acc = network.compute_accuracy(Y_train, T_train)
            acc_history.append(train_acc)

            L.test = True
            Y_test = network.forward(X_test)

            test_loss = network.compute_loss(Y_test, T_test)
            val_loss_history.append(test_loss)

            test_acc = network.compute_accuracy(Y_test, T_test)
            val_acc_history.append(test_acc)

            print('iter: %7d - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f' %(index, train_loss, train_acc, test_loss, test_acc))

    elapsed_time = time.time() - start
