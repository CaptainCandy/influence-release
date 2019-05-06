# -*- coding: utf-8 -*-
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

def fill(X, Y, idx, label, img_path, img_side):
    img = image.load_img(img_path, target_size=(img_side, img_side))
    x = image.img_to_array(img)
    X[idx, ...] = x
    Y[idx] = label

# 从原压缩包中解压并且完成自己的命名，已单独完成
def extract_and_rename_vehicles():
    class_maps = [
        ('wagon', 'n02814533'),
        ('taxi', 'n02930766'),
        ('racer', 'n04037443'),
        ('sportscar', 'n04285008')
        ]


    for class_string, class_id in class_maps:
        
        class_dir = os.path.join(BASE_DIR, class_string)
        print('class_dir = ' + class_dir)
        #call('mkdir %s' % class_dir, shell=True)
        call('tar -xf %s.tar -C %s' % (os.path.join(BASE_DIR, class_id), class_dir), shell=True)
        
        print('Decompressing %s images...' % class_string)
        for filename in os.listdir(class_dir):

            file_idx = filename.split('_')[1].split('.')[0]
            src_filename = os.path.join(class_dir, filename)
            dst_filename = os.path.join(class_dir, '%s_%s.JPEG' % (class_string, file_idx))
            os.rename(src_filename, dst_filename)

#extract_and_rename_vehicles()
# =============================================================================
# #手工修改一下类的文件名
# i=1
# class_dir = os.path.join(BASE_DIR, 'airplane')
# for filename in os.listdir(class_dir):
#     src = os.path.join(class_dir, filename)
#     dst = os.path.join(class_dir, 'airplane_%s.JPEG' % i)
#     os.rename(src, dst)
#     i += 1
# =============================================================================

def raw_images_conversion(num_train_ex_per_class=1000, 
                         num_test_ex_per_class=300
                         ):
    num_channels = 3
    img_side = 299
    
    classes = ['car', 'airplane']
    data_filename = os.path.join(BASE_DIR, 'dataset_%s_train-%s_test-%s.npz' % ('-'.join(classes), num_train_ex_per_class, num_test_ex_per_class))

    num_classes = len(classes)
    num_train_examples = num_train_ex_per_class * num_classes
    num_test_examples = num_test_ex_per_class * num_classes
    
    # =============================================================================
    #     if num_valid_ex_per_class == 0:
    #         valid_str = ''
    #     else:
    #         valid_str = '_valid-%s' % 100 #TODO: 解决valid example的问题
    # =============================================================================
    
    # =============================================================================
    #     if classes is None:
    #         classes = ['jeep', 'airliner']
    #         data_filename = os.path.join(BASE_DIR, 'dataset_train-%s_test-%s%s.npz' % (num_train_ex_per_class, num_test_ex_per_class, valid_str))
    #     else:
    #         data_filename = os.path.join(BASE_DIR, 'dataset_%s_train-%s_test-%s%s.npz' % ('-'.join(classes), num_train_ex_per_class, num_test_ex_per_class, valid_str))
    # =============================================================================
    
    print('Reading vehicles from raw images...')
    X_train = np.zeros([num_train_examples, img_side, img_side, num_channels])
    X_test = np.zeros([num_test_examples, img_side, img_side, num_channels])

    Y_train = np.zeros([num_train_examples])
    Y_test = np.zeros([num_test_examples])

    for class_idx, class_string in enumerate(classes):
        print('class: %s' % class_string)            
        # For some reason, a lot of numbers are skipped.
        i = 0
        num_filled = 0
        while num_filled < num_train_ex_per_class:        
            img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
            if os.path.exists(img_path):
                fill(X_train, Y_train, num_filled + (num_train_ex_per_class * class_idx), class_idx, img_path, img_side)
                num_filled += 1
            i += 1
        print(num_filled)

        num_filled = 0
        while num_filled < num_test_ex_per_class:        
            img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
            if os.path.exists(img_path):
                fill(X_test, Y_test, num_filled + (num_test_ex_per_class * class_idx), class_idx, img_path, img_side)
                num_filled += 1
            i += 1
        print(num_filled)
    
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    np.random.seed(0)
    permutation_idx = np.arange(num_train_examples)
    np.random.shuffle(permutation_idx)
    X_train = X_train[permutation_idx, :]
    Y_train = Y_train[permutation_idx]
    permutation_idx = np.arange(num_test_examples)
    np.random.shuffle(permutation_idx)
    X_test = X_test[permutation_idx, :]
    Y_test = Y_test[permutation_idx]
    
    np.savez_compressed(data_filename, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

#raw_images_conversion()

#遗留validation的样本数量问题没解决
def load_vehicles(num_train_ex_per_class=1000, 
                 num_test_ex_per_class=300
                 ):    
    
    classes = ['car', 'airplane']
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

