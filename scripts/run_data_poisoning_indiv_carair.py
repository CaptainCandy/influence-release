# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 08:57:24 2019

@author: Administrator
"""

import IPython
import numpy as np

#from load_animals import load_animals, load_dogfish_with_koda, load_dogfish_with_orig_and_koda
from load_vehicles import load_vehicles

import os
from shutil import copyfile
from datetime import datetime

from influence.inceptionModel import BinaryInceptionModel
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from influence.dataset import DataSet
from influence.dataset_poisoning import iterative_attack, select_examples_to_attack, get_projection_to_box_around_orig_point, generate_inception_features

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base


img_side = 299
num_channels = 3
 
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]

weight_decay = 0.001

num_classes = 2
max_lbfgs_iter = 1000

num_train_ex_per_class = 1000
num_test_ex_per_class = 300
batch_size = 100

dataset_name = 'carair_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
data_sets = load_vehicles(
    num_train_ex_per_class=num_train_ex_per_class, 
    num_test_ex_per_class=num_test_ex_per_class)


full_graph = tf.Graph()
top_graph = tf.Graph()

print('*** Full:')
with full_graph.as_default():
    full_model_name = '%s_inception_wd-%s' % (dataset_name, weight_decay)
    full_model = BinaryInceptionModel(
        img_side=img_side,
        num_channels=num_channels,
        weight_decay=weight_decay,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=True,
        train_dir='output15',
        log_dir='log',
        model_name=full_model_name)

    for data_set, label in [
        (data_sets.train, 'train'),
        (data_sets.test, 'test')]:

        inception_features_path = 'output15/%s_inception_features_new_%s.npz' % (dataset_name, label)
        if not os.path.exists(inception_features_path):

            print('Inception features do not exist. Generating %s...' % label)
            data_set.reset_batch()
            
            num_examples = data_set.num_examples
            assert num_examples % batch_size == 0

            inception_features_val = generate_inception_features(
                full_model, 
                data_set.x, 
                data_set.labels, 
                batch_size=batch_size)
            
            np.savez(
                inception_features_path, 
                inception_features_val=inception_features_val,
                labels=data_set.labels)


train_f = np.load('output15/%s_inception_features_new_train.npz' % dataset_name)
train = DataSet(train_f['inception_features_val'], train_f['labels'])
test_f = np.load('output15/%s_inception_features_new_test.npz' % dataset_name)
test = DataSet(test_f['inception_features_val'], test_f['labels'])

validation = None

inception_data_sets = base.Datasets(train=train, validation=validation, test=test)

print('*** Top:')
with top_graph.as_default():
    top_model_name = '%s_inception_onlytop_wd-%s' % (dataset_name, weight_decay)
    input_dim = 2048
    top_model = BinaryLogisticRegressionWithLBFGS(
        input_dim=input_dim,
        weight_decay=weight_decay,
        max_lbfgs_iter=max_lbfgs_iter,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=inception_data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output15',
        log_dir='log',
        model_name=top_model_name)
    top_model.train()
    weights = top_model.sess.run(top_model.weights)
    orig_weight_path = 'output15/inception_weights_%s.npy' % top_model_name
    np.save(orig_weight_path, weights)


with full_graph.as_default():
    full_model.load_weights_from_disk(orig_weight_path, do_save=False, do_check=True)
    full_model.reset_datasets()



step_size = 0.02

num_train = len(top_model.data_sets.train.labels)
num_test = len(top_model.data_sets.test.labels)
max_num_to_poison = 10
loss_type = 'normal_loss'


### Try attacking each test example individually

orig_X_train = np.copy(data_sets.train.x)
orig_Y_train = np.copy(data_sets.train.labels)
#%%
### Create poisoned dataset
print('Creating poisoned dataset...')

#for test_idx in range(0, num_test):
for test_idx in range(321, 600):

    print('****** Attacking test_idx %s ******' % test_idx)
    test_description = test_idx

    # If this has already been successfully attacked, skip
    filenames = [filename for filename in os.listdir('output15') if (
        (('%s_attack_%s_testidx-%s_trainidx-' % (full_model.model_name, loss_type, test_description)) in filename) and        
        (filename.endswith('stepsize-%s_proj_final.npz' % step_size)))]
        # and (('stepsize-%s_proj_final.npz' % step_size) in filename))] # Check all step sizes        
    if len(filenames) > 0:
        print('test_idx %s has already been successfully attacked. Skipping...')
        continue


    # Use top model to quickly generate inverse HVP
    with top_graph.as_default():
        top_model.get_influence_on_test_loss(
            [test_idx], 
            [0],        
            test_description=test_description,
            force_refresh=True)
    copyfile(
        'output15/%s-cg-normal_loss-test-%s.npz' % (top_model_name, test_description),
        'output15/%s-cg-normal_loss-test-%s.npz' % (full_model_name, test_description))

    # Use full model to select indices to poison
    start = datetime.now()
    with full_graph.as_default():
        #print("here.grad_influence_wrt_input_val.")
        grad_influence_wrt_input_val = full_model.get_grad_of_influence_wrt_input(
            np.arange(num_train), 
            [test_idx], 
            force_refresh=False,
            test_description=test_description,
            loss_type=loss_type)
        #print('here.all_indices_to_poison')
        all_indices_to_poison = select_examples_to_attack(
            full_model, 
            max_num_to_poison, 
            grad_influence_wrt_input_val,
            step_size=step_size)
    end = datetime.now()
    print('get indices to poison took %s seconds.' % (end-start).seconds)

    for num_to_poison in [0.1, 1.2, 2, 3, 5, 10]:
    #for num_to_poison in [0.1, 1.2]:
        # If we're just attacking one training example, try attacking the first one and also the second one separately
        print('attacking %s...' % num_to_poison)
        start = datetime.now()
        if num_to_poison == 0.1:
            indices_to_poison = all_indices_to_poison[0:1]
        elif num_to_poison == 1.2:
            indices_to_poison = all_indices_to_poison[1:2]
        else:
            indices_to_poison = all_indices_to_poison[:num_to_poison]
        orig_X_train_subset = np.copy(full_model.data_sets.train.x[indices_to_poison, :])
        orig_X_train_inception_features_subset = np.copy(top_model.data_sets.train.x[indices_to_poison, :])

        project_fn = get_projection_to_box_around_orig_point(orig_X_train_subset, box_radius_in_pixels=0.5)

        attack_success = iterative_attack(top_model, full_model, top_graph, full_graph, project_fn, [test_idx], test_description=test_description, 
            indices_to_poison=indices_to_poison,
            num_iter=50,
            step_size=step_size,
            save_iter=50,
            loss_type=loss_type,
            early_stop=0.5)

        with full_graph.as_default():
            full_model.update_train_x_y(orig_X_train, orig_Y_train)
            full_model.load_weights_from_disk(orig_weight_path, do_save=False, do_check=False)
        with top_graph.as_default():
            X_train = top_model.data_sets.train.x
            X_train[indices_to_poison, :] = orig_X_train_inception_features_subset
            top_model.update_train_x(X_train)
            top_model.train()
        
        end = datetime.now()
        print('attacking %s took %s seconds.' % (num_to_poison, (end-start).seconds))

        if attack_success:
            break