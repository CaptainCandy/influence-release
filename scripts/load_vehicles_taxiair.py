# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:30:26 2019

@author: Administrator
"""

# Forked from load_animals

import os

from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import IPython

from subprocess import call

from keras.preprocessing import image

import sys
sys.path.append("C:/Tang/influence-release-master")  #设置自定义包的搜索路径
from influence.dataset import DataSet
from influence.inception_v3 import preprocess_input

BASE_DIR = 'C:/Tang/data/taxi_airplane/' # TODO: change

def load_vehicles(num_train_ex_per_class=1000, 
                 num_test_ex_per_class=300
                 ):    
    
    classes = ['taxi', 'airplane']
    data_filename = os.path.join(BASE_DIR, 'dataset_%s_train-%s_test-%s.npz' % ('-'.join(classes), num_train_ex_per_class, num_test_ex_per_class))

    if os.path.exists(data_filename):
        print('Loading vehicles from disk...')
        f = np.load(data_filename)
        X_train = f['X_train']
        X_test = f['X_test']
        Y_train = f['Y_train']
        Y_test = f['Y_test']

        if 'X_valid' in f:
            X_valid = f['X_valid']
        else:
            X_valid = None

        if 'Y_valid' in f:
            Y_valid = f['Y_valid']
        else:
            Y_valid = None
    else:
        raise ValueError('No Such File.')


    train = DataSet(X_train, Y_train)
    if (X_valid is not None) and (Y_valid is not None):
        validation = DataSet(X_valid, Y_valid)
    else:
        validation = None

    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)