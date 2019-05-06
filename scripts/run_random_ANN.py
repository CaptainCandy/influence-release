# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:36:40 2019

@author: Administrator
"""

# 重新写一个随机生成权重并训练的inception v3网络

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

#%%
data_set = np.load('C:/Tang/data/car_airplane/dataset_car-airplane_train-1000_test-300.npz')
X_train = data_set['X_train'].reshape((2000, -1))
Y_train = data_set['Y_train']
X_test = data_set['X_test'].reshape((600, -1))
Y_test = data_set['Y_test']

#%%
data_set = np.load('C:/Tang/data/car_airplane/dataset_car-airplane_train-40_test-300.npz')
X_train = data_set['X_train'].reshape((80, -1))
Y_train = data_set['Y_train']
X_test = data_set['X_test'].reshape((600, -1))
Y_test = data_set['Y_test']

#%%
#model = InceptionV3(weights=None, include_top=True, classes =2)

classifier = Sequential()
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 268203))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#%%
classifier.compile(loss='binary_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

history = classifier.fit(X_train, Y_train,
                    batch_size=100,
                    epochs=25,
                    validation_data=(X_test, Y_test),
                    verbose = 2)