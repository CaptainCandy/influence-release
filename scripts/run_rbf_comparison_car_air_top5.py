# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:26:35 2019

@author: Administrator
"""
# Forked from run_rbf_comparison.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import math
import copy
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import scipy
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
import random

import sys
sys.path.append("C:/Tang/influence-release-master")  #设置自定义包的搜索路径

from load_vehicles import load_vehicles

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from sklearn.metrics.pairwise import rbf_kernel

from influence.inceptionModel import BinaryInceptionModel
from influence.smooth_hinge import SmoothHinge
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.dataset_poisoning import generate_inception_features

#%%
def get_Y_pred_correct_inception(model):
    Y_test = model.data_sets.test.labels
    if np.min(Y_test) < -0.5:
        Y_test = (np.copy(Y_test) + 1) / 2        
    Y_pred = model.sess.run(model.preds, feed_dict=model.all_test_feed_dict)
    Y_pred_correct = np.zeros([len(Y_test)])
    for idx, label in enumerate(Y_test):
        Y_pred_correct[idx] = Y_pred[idx, int(label)]
    return Y_pred_correct

num_classes = 2
num_train_ex_per_class = 40
num_test_ex_per_class = 300

dataset_name = 'carair_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
image_data_sets = load_vehicles(
    num_train_ex_per_class=num_train_ex_per_class, 
    num_test_ex_per_class=num_test_ex_per_class)

weight_decay = 0.001
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]

#%%
### Generate kernelized feature vectors
X_train = image_data_sets.train.x
X_test = image_data_sets.test.x

Y_train = np.copy(image_data_sets.train.labels) * 2 - 1
Y_test = np.copy(image_data_sets.test.labels) * 2 - 1

num_train = X_train.shape[0]
num_test = X_test.shape[0]

X_stacked = np.vstack((X_train, X_test))

gamma = 0.05
weight_decay = 0.0001

K = rbf_kernel(X_stacked, gamma = gamma / num_train)

# =============================================================================
# L = slin.cholesky(K, lower=True)
# L_train = L[:num_train, :num_train]
# L_test = L[num_train:, :num_train]
# =============================================================================

K_train = K[:num_train, :num_train]
K_test = K[num_train:, :num_train]

### Compare top 5 influential examples from each network

test_idx = 0

## RBF

input_channels = 1
weight_decay = 0.001
batch_size = num_train
initial_learning_rate = 0.001 
keep_probs = None
max_lbfgs_iter = 1000
use_bias = False
decay_epochs = [1000, 10000]

tf.reset_default_graph()

X_train = image_data_sets.train.x
Y_train = image_data_sets.train.labels * 2 - 1
train = DataSet(K_train, Y_train)
test = DataSet(K_test, Y_test)

data_sets = base.Datasets(train=train, validation=None, test=test)
input_dim = data_sets.train.x.shape[1]

# Train with hinge
print('Train rbf with hinge...')
rbf_model = SmoothHinge(
    temp=0,
    use_bias=use_bias,
    input_dim=input_dim,
    weight_decay=weight_decay,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output7',
    log_dir='log',
    model_name='carair_rbf_hinge_t-0')
    
rbf_model.train()
hinge_W = rbf_model.sess.run(rbf_model.params)[0]

# Then load weights into smoothed version
print('Load weights into smoothed version...')
tf.reset_default_graph()
rbf_model = SmoothHinge(
    temp=0.001,
    use_bias=use_bias,
    input_dim=input_dim,
    weight_decay=weight_decay,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output7',
    log_dir='log',
    model_name='car_air_rbf_hinge_t-0.001')

params_feed_dict = {}
params_feed_dict[rbf_model.W_placeholder] = hinge_W
rbf_model.sess.run(rbf_model.set_params_op, feed_dict=params_feed_dict)

rbf_predicted_loss_diffs = rbf_model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(rbf_model.data_sets.train.labels)),
    force_refresh=True)

#%%
## Inception

dataset_name = 'carair_40_300'
test_idx = 0

# Generate inception features
print('Generate inception features...')
img_side = 299
num_channels = 3
num_train_ex_per_class = 40
num_test_ex_per_class = 300
batch_size = 20 #TODO: 需要根据配置修改

# reset_default_graph大概就是重置当前线程，让tf session里定义的东西都失效，重来。就是重开一个神经网络session
tf.reset_default_graph()
full_model_name = '%s_inception' % dataset_name
# 下面的语句是定义一个inception双分类器
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
    train_dir='output9',
    log_dir='log',
    model_name=full_model_name)

# 下面的代码是在使用inception的卷积层生成特征
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

train = DataSet(
    train_inception_features_val,
    image_data_sets.train.labels)
test = DataSet(
    test_inception_features_val,
    image_data_sets.test.labels)

# train_f = np.load('G:/output/%s_inception_features_new_train.npz' % dataset_name)
# train = DataSet(train_f['inception_features_val'], train_f['labels'])
# test_f = np.load('G:/output/%s_inception_features_new_test.npz' % dataset_name)
# test = DataSet(test_f['inception_features_val'], test_f['labels'])

validation = None

# 上面的代码是训练了inception模型的全连接层前面的部分，因此输出的feature有2048个维度
data_sets = base.Datasets(train=train, validation=validation, test=test)

# train_f = np.load('G:/output/%s_inception_features_new_train.npz' % dataset_name)
# train = DataSet(train_f['inception_features_val'], train_f['labels'])
# test_f = np.load('G:/output/%s_inception_features_new_test.npz' % dataset_name)
# test = DataSet(test_f['inception_features_val'], test_f['labels'])
# validation = None

# data_sets = base.Datasets(train=train, validation=validation, test=test)

# 下面的代码利用从inception卷积层训练完成后的feature进行一个二分类逻辑回归，取消卷积层后面的FC全连接层
print('Train logistic regression after inception...')
input_dim = 2048
weight_decay = 0.001
batch_size = 20
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000
num_classes = 2

tf.reset_default_graph()

inception_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output9',
    log_dir='log',
    model_name='%s_inception_onlytop' % dataset_name)

inception_model.train()

# =============================================================================
# inception_predicted_loss_diffs = inception_model.get_influence_on_test_loss(
#     [test_idx], 
#     np.arange(len(inception_model.data_sets.train.labels)),
#     force_refresh=True)
# 
# x_test = X_test[test_idx, :]
# y_test = Y_test[test_idx]
# 
# 
# distances = dataset.find_distances(x_test, X_train)
# flipped_idx = Y_train != y_test
# rbf_margins_test = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_test_feed_dict)
# rbf_margins_train = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_train_feed_dict)
# inception_Y_pred_correct = get_Y_pred_correct_inception(inception_model)
# 
# 
# np.savez(
#     'output7/rbf_carair_results_%s' % test_idx,
#     test_idx=test_idx,
#     distances=distances,
#     flipped_idx=flipped_idx,
#     rbf_margins_test=rbf_margins_test,
#     rbf_margins_train=rbf_margins_train,
#     inception_Y_pred_correct=inception_Y_pred_correct,
#     rbf_predicted_loss_diffs=rbf_predicted_loss_diffs,
#     inception_predicted_loss_diffs=inception_predicted_loss_diffs
# )
# =============================================================================

#%%
print('Save results...')
#rand_test = random.sample(range(1, 600),50)
#np.savez('output7/rand_test_point', rand_test=rand_test)
for test_idx in range(1, 600):
    rbf_predicted_loss_diffs = rbf_model.get_influence_on_test_loss(
        [test_idx], 
        np.arange(len(rbf_model.data_sets.train.labels)),
        force_refresh=True)
    inception_predicted_loss_diffs = inception_model.get_influence_on_test_loss(
        [test_idx], 
        np.arange(len(inception_model.data_sets.train.labels)),
        force_refresh=True)
    x_test = X_test[test_idx, :]
    y_test = Y_test[test_idx]
    
    distances = dataset.find_distances(x_test, X_train)
    flipped_idx = Y_train != y_test
    rbf_margins_test = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_test_feed_dict)
    rbf_margins_train = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_train_feed_dict)
    inception_Y_pred_correct = get_Y_pred_correct_inception(inception_model)
    
    np.savez(
        'output9/rbf_carair_results_%s' % test_idx,
        test_idx=test_idx,
        distances=distances,
        flipped_idx=flipped_idx,
        rbf_margins_test=rbf_margins_test,
        rbf_margins_train=rbf_margins_train,
        inception_Y_pred_correct=inception_Y_pred_correct,
        rbf_predicted_loss_diffs=rbf_predicted_loss_diffs,
        inception_predicted_loss_diffs=inception_predicted_loss_diffs
    )
