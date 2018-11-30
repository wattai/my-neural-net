#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:24:04 2017

@author: wattai
"""

import sys
from chainer import cuda

import optimizers
import layers as L
import utilities as ut
import initializers as init


class Model:
    def __init__(self, lossfunc, optimizer, batch_size=32,):
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.batch_size = batch_size

        input_size = 64
        hidden_size = 3136
        output_size = 10

        # self.lr = 0.001
        # self.alpha = 0.9
        self.l1 = 1e-4
        self.l2 = 1e-4
        self.optimizer = optimizers.Adam(l1=self.l1, l2=self.l2)

        self.conv0 = L.Convolution_(n_filter=8, filter_size=(3, 3), stride=1)
        self.conv1 = L.Convolution_(n_filter=16, filter_size=(3, 3), stride=1)
        self.conv2 = L.Convolution_(n_filter=32, filter_size=(5, 5), stride=1)
        self.conv3 = L.Convolution_(n_filter=64, filter_size=(5, 5), stride=1)

        self.fc0 = L.Linear_(output_size=1024)
        self.fc1 = L.Linear_(output_size=10)

        self.bn0 = L.BatchNormalization_()
        self.bn1 = L.BatchNormalization_()
        self.bn2 = L.BatchNormalization_()
        self.bn3 = L.BatchNormalization_()
        self.bn4 = L.BatchNormalization_()

        self.acti0 = L.ELU()
        self.acti1 = L.ELU()
        self.acti2 = L.ELU()
        self.acti3 = L.ELU()
        self.acti4 = L.ELU()

        self.pool0 = L.MaxPooling(7, 7)
        self.pool1 = L.MaxPooling(5, 5)
        self.pool2 = L.MaxPooling(3, 3)
        self.pool3 = L.MaxPooling(3, 3)

        self.flat = L.Flatten()

        self.drop0 = L.Dropout(0.5)
        self.drop1 = L.Dropout(0.5)
        self.drop2 = L.Dropout(0.5)
        self.drop3 = L.Dropout(0.25)
        
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
                       
                       #self.conv2,
                       #self.acti2,
                       #self.pool2,
                       #self.bn2,
                       #self.drop2,
                       
                       #self.conv3,
                       #self.acti3,
                       #self.pool3,
                       #self.bn3,
                       #self.drop3,
                       
                       self.flat,
                       
                       self.fc0,
                       self.acti4,
                       self.bn4,
                       
                       self.fc1,
                       ]


    def forward(self, x):
        xp = cuda.get_array_module(x)
        if x.shape[0] > self.batch_size:
            y = xp.zeros(  # creation of output array.
                    tuple([x.shape[0]]) +
                    self.forward(x[0][None, :]).shape[1:]
                )
            for i in range(0, x.shape[0]-self.batch_size+1, self.batch_size):
                if len(y[i:]) >= self.batch_size:
                    y[i:i+self.batch_size] = self.forward(
                            x[i:i+self.batch_size])
                else:
                    y[i:] = self.forward(x[i:])
            return y
        else:
            for layer in self.layers:
                x = layer.forward(x)
            return x

    def backward(self, x, t):
        xp = cuda.get_array_module(x)
        self.loss = self.lossfunc.forward(x, t)
        dy = xp.array([1.0])
        grads = {}
        for layer in ([self.lossfunc] + self.layers[::-1]):
            dy = layer.backward(dy)
            for key in layer.grads.keys():
                grads[key] = layer.grads[key]
        return grads

    def update(self, x, t):
        xp = cuda.get_array_module(x)
        L.test = False
        self.y = self.forward(x)
        self.lossfunc.forward(self.y, t)

        params = {}
        for layer in (self.layers + [self.lossfunc]):
            for key in layer.params.keys():
                params[key] = layer.params[key]

        L.params = params
        grads = self.backward(self.y, t)

        if self.lossfunc.l1 is not None:
            self.l1 = self.lossfunc.l1
            self.optimizer.l1 = xp.exp(self.l1)
        if self.lossfunc.l2 is not None:
            self.l2 = self.lossfunc.l2
            self.optimizer.l2 = xp.exp(self.l2)
        # print('exp(l1): %f' % xp.exp(self.l1),
        #       'exp(l2): %f' % xp.exp(self.l2))

        self.params = self.optimizer.update(params, grads)

    def compute_loss(self, y, t):
        if y.shape[0] != t.shape[0]:
            print('Array size is NOT match.')
            sys.exit(1)
            return None

        loss_l1 = ut.l1_norm(self.params)
        loss_l2 = ut.l2_norm(self.params)
        loss = 0.0
        for i in range(0, y.shape[0]-self.batch_size+1, self.batch_size):
            if len(y[i:]) >= self.batch_size:
                loss += self.compute_loss_sum(y[i:i+self.batch_size],
                                              t[i:i+self.batch_size])
            else:
                loss += self.compute_loss_sum(y[i:], t[i:])
        return (loss / y.shape[0]) + (
                self.optimizer.l1 * loss_l1 + self.optimizer.l2 * loss_l2)

    def compute_loss_sum(self, y, t):
        xp = cuda.get_array_module(y)
        return xp.sum(self.lossfunc.forward(y, t), axis=0)

    def compute_accuracy(self, y, t):
        if y.shape[0] != t.shape[0]:
            print('Array size is NOT match.')
            sys.exit(1)
            return None

        acc = 0.0
        for i in range(0, y.shape[0]-self.batch_size+1, self.batch_size):
            if len(y[i:]) >= self.batch_size:
                acc += self.compute_accuracy_sum(y[i:i+self.batch_size],
                                                 t[i:i+self.batch_size])
            else:
                acc += self.compute_accuracy_sum(y[i:], t[i:])
        return acc / y.shape[0]

    def compute_accuracy_sum(self, y, t):
        xp = cuda.get_array_module(y)
        return xp.sum(ut.softmax(y).argmax(axis=1) == t.argmax(axis=1), axis=0)
