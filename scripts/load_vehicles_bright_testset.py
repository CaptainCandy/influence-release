# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:05:10 2019

@author: Tang
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

BASE_DIR = 'C:/Tang/data/car_airplane/' # TODO: change

def load_vehicles_bright_testset(num_train_ex_per_class=1000, 
                 num_test_ex_per_class=300
                 ):    
    
    classes = ['car', 'airplane']
    data_filename = os.path.join(BASE_DIR, 'dataset_%s_train-%s_test-%s.npz' % ('-'.join(classes), num_train_ex_per_class, num_test_ex_per_class))

    if os.path.exists(data_filename):
        print('Loading vehicles from disk...')
        f = np.load(data_filename)
        f_bright = np.load(BASE_DIR + 'dataset_car-airplane_bright_testset.npz')
        X_train = f['X_train']
        X_test = f_bright['X_test']
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