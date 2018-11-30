# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:36:24 2018

@author: wattai
"""

# import cupy as xp
from chainer import cuda

import utilities as ut
import initializers as init

test = False
names = []
params = None


class Linear:
    def __init__(self, W, b):
        self.W = W  # .astype(np.float32)
        self.b = b  # .astype(np.float32)
        self.x = None

        self.dW = None
        self.db = None

        self.xp = None

        name = 'lin'
        self.name = ut.naming(name, names)
        names.append(self.name)

#        self.params = {}
#        self.params['W_' + self.name] = self.W
#        self.params['b_' + self.name] = self.b
        self.grads = {}
        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db

    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        self.x = x
        return self.xp.dot(self.x, self.W) + self.b

    def backward(self, dy):
        dx = self.xp.dot(dy, self.W.T)
        self.dW = self.xp.dot(self.x.T, dy)
        self.db = self.xp.sum(dy, axis=0)

        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db
        return dx


class Linear_:
    def __init__(self, output_size):
        self.output_size = output_size

        self.W = None#.astype(np.float32)
        self.b = None#.astype(np.float32)
        self.x = None
        
        self.dW = None
        self.db = None
        
        self.xp = None
        
        name = 'lin'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.params['W_' + self.name] = self.W
        self.params['b_' + self.name] = self.b
        self.grads = {}
        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db

    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        self.x = x

        if self.W is None and self.b is None:
            self.W = self.xp.array(init.he_normal([x.shape[1], self.output_size]))
            self.b = self.xp.array(init.he_normal([self.output_size]))
        
        self.params['W_' + self.name] = self.W
        self.params['b_' + self.name] = self.b
        return self.xp.dot(self.x, self.W) + self.b
    
    def backward(self, dy):
        dx = self.xp.dot(dy, self.W.T)
        self.dW = self.xp.dot(self.x.T, dy)
        self.db = self.xp.sum(dy, axis=0)
        
        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W#.astype(np.float32)
        self.b = b#.astype(np.float32)
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None
        
        self.xp = None
        
        name = 'conv'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
#        self.params = {}
#        self.params['W_' + self.name] = self.W
#        self.params['b_' + self.name] = self.b
        self.grads = {}
        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db

    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        C, FN, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = ut.im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = self.xp.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) + self.b

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        C, FN, FH, FW = self.W.shape
        
        self.db = self.xp.sum(dout, axis=0)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.dW = self.xp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(C, FN, FH, FW)

        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db

        dcol = self.xp.dot(dout, self.col_W.T)
        dx = ut.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Convolution_:
    def __init__(self, n_filter, filter_size, stride=1, pad=0):
        self.FN = n_filter
        self.FH = filter_size[0]
        self.FW = filter_size[1]
        
        self.W = None#.astype(np.float32)
        self.b = None#.astype(np.float32)
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None
        
        self.xp = None
        
        name = 'conv'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.params['W_' + self.name] = self.W
        self.params['b_' + self.name] = self.b
        self.grads = {}
        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db

    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        #C, FN, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - self.FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - self.FW) / self.stride)
        
        if self.W is None and self.b is None:
            self.W = self.xp.array(init.he_normal([C, self.FN, self.FH, self.FW]))
            self.b = self.xp.array(init.he_normal([self.FN, out_h, out_w]))
            
        col = ut.im2col(x, self.FH, self.FW, self.stride, self.pad)
        col_W = self.W.reshape(self.FN, -1).T

        out = self.xp.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        
        self.params['W_' + self.name] = self.W
        self.params['b_' + self.name] = self.b
        return out

    def backward(self, dout):
        C, FN, FH, FW = self.W.shape
        
        self.db = self.xp.sum(dout, axis=0)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.dW = self.xp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(C, FN, FH, FW)

        self.grads['W_' + self.name] = self.dW
        self.grads['b_' + self.name] = self.db

        dcol = self.xp.dot(dout, self.col_W.T)
        dx = ut.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        
        self.xp = None
        
        name = 'MaxPooling'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}

    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = ut.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = self.xp.argmax(col, axis=1)
        out = self.xp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = self.xp.zeros((dout.size, pool_size)).astype(self.xp.float32)
        dmax[self.xp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = ut.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
        
class ReLU():
    def __init__(self):
        self.mask = None
        
        name = 'ReLU'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}
        
    def forward(self, x):
        self.mask = (x < 0)
        y = x.copy()
        y[self.mask] = 0
        return y
    
    def backward(self, dy):
        dx = dy.copy()
        dx[self.mask] = 0
        return dx
    
    
class ELU():
    def __init__(self, alpha=1.0):
        self.mask = None
        self.alpha = alpha
        self.y = None
        
        self.xp = None
        
        name = 'ELU'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}
        
    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        self.mask = (x < 0)
        self.y = x.copy()
        self.y[self.mask] = (self.alpha * (self.xp.expm1(x)))[self.mask]
        return self.y
    
    def backward(self, dy):
        dx = dy.copy()
        dx[self.mask] = ((self.y + self.alpha) * dy)[self.mask]
        return dx
    

class SELU():
    def __init__(self, alpha=1.6732632423543772, scale=1.0507009873554805):
        self.mask = None
        self.y = None
        self.alpha = alpha
        self.scale = scale
        self.dalpha = None
        self.dscale = None
        
        self.xp = None
        
        name = 'selu'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
#        self.params = {}
#        self.params['a_' + self.name] = self.alpha
#        self.params['s_' + self.name] = self.scale
        self.grads = {}
        self.grads['a_' + self.name] = self.dalpha
        self.grads['s_' + self.name] = self.dscale
        
    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        self.alpha = self.xp.array([self.alpha]).squeeze()
        self.scale = self.xp.array([self.scale]).squeeze()
        
        self.mask = (x < 0)
        self.y = self.scale * x.copy()
        self.y[self.mask] = (self.scale * self.alpha * (self.xp.expm1(x)))[self.mask]
        return self.y
    
    def backward(self, dy):
        #self.dalpha = xp.zeros_like(self.alpha)
        #self.dscale = xp.zeros_like(self.scale)
        
        dx = self.scale * dy.copy()
        dx[self.mask] = ((self.y + self.scale * self.alpha) * dy)[self.mask]
        self.dalpha = self.xp.sum(self.y[self.mask] * dy[self.mask]) / self.alpha
        self.dscale = self.xp.sum((self.xp.log(self.y / self.scale / self.alpha + 1) * dy)[~self.mask])
        self.dscale += self.xp.sum((self.y * dy)[self.mask]) / self.scale
        return dx
    

class Flatten():
    def __init__(self):
        self.x_shape = None
        
        name = 'Flatten'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}
        
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dy):
        return dy.reshape(self.x_shape)
    

class BatchNormalization():
    def __init__(self, gamma, beta):
        self.eps = 1e-5
        self.gamma = gamma
        self.beta = beta
        self.dgamma = None
        self.dbeta = None
        
        self.xp = None
        
        name = 'bn'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
#        self.params = {}
#        self.params['g_' + self.name] = self.gamma
#        self.params['b_' + self.name] = self.beta
        self.grads = {}
        self.grads['g_' + self.name] = self.dgamma
        self.grads['b_' + self.name] = self.dbeta
        
    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        x_shape = x.shape
        ex_dims = len(x_shape) - 2
        self.axes = tuple(self.xp.concatenate((self.xp.array([0]),
                                               self.xp.arange(2, 2+ex_dims))).astype(self.xp.int32).tolist())
        
        cast_shape = self.xp.concatenate((self.xp.array([1, x_shape[1]]),
                                          self.xp.ones(ex_dims)), axis=0).astype(self.xp.int32).tolist()
        self.gamma = self.gamma.reshape(cast_shape)
        self.beta = self.beta.reshape(cast_shape)      
        
        self.x_mu = x - self.xp.mean(x, axis=self.axes, keepdims=True)
        self.sqrtvar = self.xp.sqrt(self.xp.var(x, axis=self.axes, keepdims=True) + self.eps)
        self.x_hat = self.x_mu / self.sqrtvar        
        return self.gamma * self.x_hat + self.beta

    def backward(self, dy):
        self.dgamma = self.xp.sum(dy * self.x_hat, axis=self.axes)
        self.dbeta = self.xp.sum(dy, axis=self.axes)
        self.grads['g_' + self.name] = self.dgamma
        self.grads['b_' + self.name] = self.dbeta
        
        dx_hat = self.gamma * dy / self.sqrtvar
        divar = self.xp.mean(self.x_mu * dx_hat, axis=self.axes, keepdims=True)
        dx_mu = dx_hat - self.x_mu * divar / (self.sqrtvar**(3/2))
        return dx_mu - self.xp.mean(dx_mu, axis=self.axes, keepdims=True)
    

class BatchNormalization_():
    def __init__(self):
        self.eps = 1e-5
        self.gamma = None
        self.beta = None
        self.dgamma = None
        self.dbeta = None
        
        self.xp = None
        
        name = 'bn'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.params['g_' + self.name] = self.gamma
        self.params['b_' + self.name] = self.beta
        self.grads = {}
        self.grads['g_' + self.name] = self.dgamma
        self.grads['b_' + self.name] = self.dbeta
        
    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        x_shape = x.shape
        ex_dims = len(x_shape) - 2
        self.axes = tuple(self.xp.concatenate((self.xp.array([0]),
                                               self.xp.arange(2, 2+ex_dims))).astype(self.xp.int32).tolist())
        
        cast_shape = self.xp.concatenate((self.xp.array([1, x_shape[1]]),
                                          self.xp.ones(ex_dims)), axis=0).astype(self.xp.int32).tolist()
        
        if self.gamma is None and self.beta is None:
            self.gamma = self.xp.ones(cast_shape)
            self.beta = self.xp.zeros(cast_shape) 
        
        self.x_mu = x - self.xp.mean(x, axis=self.axes, keepdims=True)
        self.sqrtvar = self.xp.sqrt(self.xp.var(x, axis=self.axes, keepdims=True) + self.eps)
        self.x_hat = self.x_mu / self.sqrtvar
        
        self.params['g_' + self.name] = self.gamma
        self.params['b_' + self.name] = self.beta
        return self.gamma * self.x_hat + self.beta

    def backward(self, dy):
        self.dgamma = self.xp.sum(dy * self.x_hat, axis=self.axes, keepdims=True)
        self.dbeta = self.xp.sum(dy, axis=self.axes, keepdims=True)
        self.grads['g_' + self.name] = self.dgamma
        self.grads['b_' + self.name] = self.dbeta
        
        dx_hat = self.gamma * dy / self.sqrtvar
        divar = self.xp.mean(self.x_mu * dx_hat, axis=self.axes, keepdims=True)
        dx_mu = dx_hat - self.x_mu * divar / (self.sqrtvar**(3/2))
        return dx_mu - self.xp.mean(dx_mu, axis=self.axes, keepdims=True)
    
    
class Dropout():
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.mask = None
        
        self.xp = None
        
        name = 'Dropout'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}
        
    def forward(self, x):
        self.xp = cuda.get_array_module(x)
        global test
        if test == False:
            """
            self.mask = np.random.binomial(n = 1,
                                           p = 1 - self.ratio,
                                           size = x.shape[1:]).astype(np.float32)
            """
            self.mask = self.ratio <= self.xp.random.uniform(low=0, high=1, size=x.shape[1:])
            return self.mask * x
        else:
            return (1. - self.ratio) * x # weight scaled.
        
    def backward(self, dy):
        return self.mask * dy


class Softmax_cross_entropy:
    def __init__(self):
        self.y = None
        self.t = None
        
        name = 'Softmax_cross_entropy'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}
    
    def forward(self, x, t):
        self.t = t
        self.y = ut.softmax(x)
        return ut.softmax_cross_entropy(x, t)
    
    def backward(self, dy):
        return (self.y - self.t) / self.t.shape[0]
    

class Softmax_cross_entropy_with_variational_regularizer:
    def __init__(self, variational=False):
        self.y = None
        self.t = None
        self.l1 = None
        self.l2 = None
        self.dl1 = None
        self.dl2 = None
        self.variational = variational

        name = 'Softmax_cross_entropy'
        self.name = ut.naming(name, names)
        names.append(self.name)

        self.params = {}
        self.grads = {}
        self.params['l1_' + self.name] = 0  # self.l1
        self.params['l2_' + self.name] = 0  # self.l2
        self.grads['l1_' + self.name] = 0  # self.dl1
        self.grads['l2_' + self.name] = 0  # self.dl2

    def forward(self, x, t):
        if self.l1 is None and self.l2 is None:
            xp = cuda.get_array_module(x)
            self.l1 = xp.array([-5.0]).astype(xp.float32)
            self.l2 = xp.array([-5.0]).astype(xp.float32)

        self.t = t
        self.y = ut.softmax(x)
        self.E = ut.softmax_cross_entropy(x, t)
        E_hat = self.E  # / (xp.exp(self.l1) + xp.exp(self.l2))

        if self.variational is True:
            self.params['l1_' + self.name] = self.l1
            self.params['l2_' + self.name] = self.l2
        return E_hat

    def backward(self, dy):
        xp = cuda.get_array_module(dy)
        eps = 1e-8
        dE = (self.y - self.t) / self.t.shape[0]
        l1_norm = ut.l1_norm(params)
        l2_norm = ut.l2_norm(params)
        # print('l1_norm: %f' % l1_norm, 'l2_norm: %f' % l2_norm)
        self.dl1 = xp.sum(- xp.exp(self.l1) * self.E / (
                (xp.exp(self.l1) + xp.exp(self.l2))**2 + eps)
                ) + xp.exp(self.l1) * l1_norm
        self.dl2 = xp.sum(- xp.exp(self.l2) * self.E / (
                (xp.exp(self.l1) + xp.exp(self.l2))**2 + eps)
                ) + xp.exp(self.l2) * l2_norm

        if self.variational is True:
            self.l1 -= 0.0000001 * self.dl1 * xp.sqrt(l1_norm + eps)
            self.l2 -= 0.000001 * self.dl2 * xp.sqrt(l2_norm + eps)

        # self.grads['l1_' + self.name] = self.dl1 #/ l1_norm
        # self.grads['l2_' + self.name] = self.dl2 #/ l2_norm
        return dE  # / (self.l1**2 + self.l2**2)


class Mean_squared_error:
    def __init__(self):
        self.y = None
        self.t = None
        
        name = 'Mean_squared_error'
        self.name = ut.naming(name, names)
        names.append(self.name)
        
        self.params = {}
        self.grads = {}
    
    def forward(self, x, t):
        self.t = t
        self.y = x
        return ut.mean_squared_error(x, t)
    
    def backward(self, dy):
        return (self.y - self.t) / self.t.shape[0]
