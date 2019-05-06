# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:37:03 2019

@author: Tang
"""

# Forked from load_animals

import os

from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np

import sys
sys.path.append("C:/Tang/influence-release-master")  #设置自定义包的搜索路径
from influence.dataset import DataSet

BASE_DIR = 'C:/Tang/data/car_airplane/' # TODO: change

def load_vehicles_special():    
    
    data_filename = BASE_DIR + 'dataset_car-airplane_train-1000_test-300.npz'

    if os.path.exists(data_filename):
        print('Loading special vehicles from disk...')
        f = np.load(data_filename)
        f_special = np.load(BASE_DIR + 'dataset_special_testset.npz')
        X_train = f['X_train']
        X_test = f_special['X']
        Y_train = f['Y_train']
        Y_test = f_special['Y']

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