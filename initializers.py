# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:46:24 2018

@author: wattai
"""

import numpy as np

def he_normal(shape):
    return (np.random.normal(scale=1.0, size=shape) \
            * np.sqrt(2.0 / shape[0])).astype(np.float32)
