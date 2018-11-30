# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:33:31 2018

@author: wattai
"""

from chainer import cuda
import utilities as ut


class MomentumSGD:
    def __init__(self, lr=0.01, alpha=0.9, eps=1e-8, l1=0.0, l2=0.0):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.l1 = l1
        self.l2 = l2
        self.t = 1
        self.d = {}

    def update(self, params, grads):
        if self.t == 1:
            for key in params.keys():
                self.d[key] = 0.0 * params[key]

        for key in params.keys():
            grads[key] += grads_regulator(params, key, self.l1, self.l2)

            self.d[key] = -self.lr * (1. - self.alpha) * grads[key] \
                          +self.alpha * self.d[key]

            params[key] += self.d[key]

        self.t += 1
        return params


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, l1=0.0, l2=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l1 = l1
        self.l2 = l2
        self.t = 1
        self.d = {}

        self.m = {}
        self.v = {}

        self.xp = None

    def update(self, params, grads):
        if self.t == 1:
            for key in params.keys():
                self.m[key] = 0.0 * params[key]
                self.v[key] = 0.0 * params[key]
            # self.xp = cuda.get_array_module(params[key])

        for key in params.keys():
            # print(params[key].shape, grads[key].shape)
            grads[key] += grads_regulator(params, key, self.l1, self.l2)

            self.m[key] = self.beta1 * self.m[key] + (
                    (1. - self.beta1) * grads[key])
            self.v[key] = self.beta2 * self.v[key] + (
                    (1. - self.beta2) * (grads[key]**2))

            m_hat = self.m[key] / (1. - self.beta1**self.t)
            v_hat = self.v[key] / (1. - self.beta2**self.t)

            self.d[key] = -self.lr * m_hat / ((v_hat)**(1/2) + self.eps)

            params[key] += self.d[key]

        self.t += 1
        return params


def grads_regulator(params, key, l1=0.0, l2=0.0, eps=1e-8):
    xp = cuda.get_array_module(params[key])
    if 'W_' in key:
        return (l1 * xp.sign(params[key])
                + l2 * params[key] / xp.sqrt(xp.sum(params[key]**2) + eps))
    else:
        return 0.
