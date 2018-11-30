# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:40:54 2018

@author: wattai
"""

from chainer import cuda

def l1_norm(params):
    norm = 0.
    for key in params.keys():
        if 'W_' in key:
            xp = cuda.get_array_module(params[key])
            norm += xp.sum(xp.abs(params[key]))
    return norm

def l2_norm(params):
    norm = 0.
    for key in params.keys():
        if 'W_' in key:
            xp = cuda.get_array_module(params[key])
            norm += xp.sqrt(xp.sum(params[key]**2))
    return norm
    
def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def softmax(x):
    xp = cuda.get_array_module(x)
    exp_tmp = xp.exp(x - xp.max(x)) # prevent overflow
    if exp_tmp.ndim == 1:
        return exp_tmp / (xp.sum(exp_tmp, axis=0))
    else: # ndim == 2
        return exp_tmp / (xp.sum(exp_tmp, axis=1, keepdims=True))

def softmax_cross_entropy(y, t):
    xp = cuda.get_array_module(y)
    return -xp.sum(t * xp.log(softmax(y)), axis=1)

def mean_squared_error(y, t):
    xp = cuda.get_array_module(y)
    return xp.mean((y - t)**2, axis=1) / 2.

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """
    xp = cuda.get_array_module(input_data)
    
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = xp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w)).astype(xp.float32)

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    xp = cuda.get_array_module(col)
    
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1)).astype(xp.float32)
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

"""
def get_im2col_indices(x_shape, field_height, field_width, stride=1, padding=0):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    #assert (H + 2 * padding - field_height) % stride == 0
    #assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), np.int32(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (np.int32(k), np.int32(i), np.int32(j))
"""

def convolve_nd(x, w, s=None, axes=(-2, -1), mode='valid'):
    xp = cuda.get_array_module(x)
    if s is None:
        s = x.shape[-2:]
    x_ft = xp.fft.fftn(x, s=s, axes=axes)
    w_ft = xp.fft.fftn(w, s=s, axes=axes)
    return xp.fft.ifftn(x_ft * w_ft, axes=axes).real.astype(xp.float32)

def naming(name, names):
    i = 0
    while name + '%d'%i in names:
        i += 1
    return name + '%d'%i
