# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:03:58 2019

@author: Tang
"""

# Forked from run_rbf_comparison.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np

import sys
sys.path.append("C:/Tang/influence-release-master")  #设置自定义包的搜索路径

from load_vehicles import load_vehicles

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base


from influence.inceptionModel import BinaryInceptionModel
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.dataset_poisoning import generate_inception_features

#%%

num_classes = 2
num_train_ex_per_class = 1000
num_test_ex_per_class = 300

dataset_name = 'carair_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
image_data_sets = load_vehicles(
    num_train_ex_per_class=num_train_ex_per_class, 
    num_test_ex_per_class=num_test_ex_per_class)

initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]


### Generate kernelized feature vectors
X_train = image_data_sets.train.x
X_test = image_data_sets.test.x

Y_train = np.copy(image_data_sets.train.labels) * 2 - 1
Y_test = np.copy(image_data_sets.test.labels) * 2 - 1

num_train = X_train.shape[0]
num_test = X_test.shape[0]

X_stacked = np.vstack((X_train, X_test))

weight_decay = 0.0001

### Compare top 5 influential examples from each network

## Inception

dataset_name = 'carair_1000_300'
# test_idx = 0

# Generate inception features
print('Generate inception features...')
img_side = 299
num_channels = 3
batch_size = 100 #TODO: 需要根据设备修改的


tf.reset_default_graph()
full_model_name = '%s_inception' % dataset_name
full_model = BinaryInceptionModel(
    img_side=img_side,
    num_channels=num_channels,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=image_data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output17',
    log_dir='log',
    model_name=full_model_name)

train_inception_features_val = generate_inception_features(
    full_model, 
    image_data_sets.train.x, 
    image_data_sets.train.labels, 
    batch_size=batch_size)        
test_inception_features_val = generate_inception_features(
    full_model, 
    image_data_sets.test.x, 
    image_data_sets.test.labels, 
    batch_size=batch_size)  

#%%
BASE_DIR = 'output6/'
infl_res = np.load(BASE_DIR + 'inception_influence_result_carair_top5.npz')['result']
index = np.where(infl_res[:, 5] > 0)[0]
print(index)

distance = np.zeros((2000, 2))

for j in range(2000):
    distance[j, 1] = np.linalg.norm(train_inception_features_val[index[0]] - train_inception_features_val[j])
    if j in index:
        distance[j, 0] = 1

#%%
sum_top5 = 0.
sum_no_top5 = 0.
for i in range(2000):
    if distance[i, 0] == 1:
        sum_top5 += distance[i, 1]
    else:
        sum_no_top5 += distance[i, 1]

print(sum_top5/78, sum_no_top5/(2000-78))

#%%
np.savez('output17/features', 
         train_inception_features = train_inception_features_val,
         test_inception_features = test_inception_features_val)