# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:01:14 2019

@author: Administrator
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import sys
sys.path.append("C:/Tang/influence-release-master")  #设置自定义包的搜索路径

import gc
from numba import jit
import datetime
import numpy as np
from influence.inception_v3 import InceptionV3
#from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras import backend as K
from keras.models import load_model

from matplotlib import pyplot as plt

#%%
# 直接读原始数据集就完事儿了
DATA_DIR = 'C:/Tang/data/car_airplane/'
clean_trainset = np.load(DATA_DIR + 'dataset_car-airplane_train-1000_test-300.npz')
#X_train = clean_trainset['X_train']
#Y_train = to_categorical(clean_trainset['Y_train'], 2)
X_test = clean_trainset['X_test']
Y_test = to_categorical(clean_trainset['Y_test'], 2)

#%%
# 读取攻击后的测试集
DATA_DIR = 'C:/Tang/data/car_airplane/'
hacked_trainset = np.load(DATA_DIR + 'dataset_car-airplane_adversarial_testset_90per.npz')
X_test = hacked_trainset['X_test_adver']
Y_test = to_categorical(hacked_trainset['Y_test'], 2)

#%%
print('Generate model of inception v3 + binary softmax...')
# 产生特征，不分类
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 直接添加二分类器，此时softmax相当于逻辑回归
predictions = Dense(2, activation='softmax')(x)

# 特征+分类器的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 我们只训练顶部的分类器
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 自定义一个最优化器，先用sgd看看
sgd = optimizers.SGD(lr=0.001, 
                     decay=0.001)

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer=sgd, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%
# 在新的数据集上训练几代
print('Train inception v3 + binary softmax...')
model.fit(x=X_train, 
          y=Y_train,
          batch_size=100, 
          verbose=2,
          epochs=80)
model.save('output11/output11model.h5')

#%%
# 读取之前训练好的模型
model = load_model('output11/output11model.h5')

#%%
print('Evaluate test dataset...')
model.evaluate(x=X_test,
               y=Y_test,
               verbose=1)

#%%
print('Try to get grads...')
# 100是飞机,label=1
ori_img = X_test[100]
# Add a 4th dimension for batch size (as Keras expects)
ori_img = np.expand_dims(ori_img, axis=0)
hack_img = np.copy(ori_img)

# 定义上限
max_change_above = ori_img + 0.01
max_change_below = ori_img - 0.01

# 需要破坏成0
fake_cata = 0

model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

# 取损失函数和梯度
cost_function = model_output_layer[0, fake_cata]
gradient_function = K.gradients(cost_function, model_input_layer)[0]
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], 
                                                [cost_function, gradient_function])

#%%
print('Try to use FGSM on test point...')
e = 0.008
cost = 0.0
hack_round = 1
while cost < 0.90:
   cost, gradients = grab_cost_and_gradients_from_model([hack_img, 0])
   g = np.sign(gradients)
   hack_img += g*e
   hack_img = np.clip(hack_img, max_change_below, max_change_above)
   hack_img = np.clip(hack_img, -1.0, 1.0)
   print("hack_round:{} prob: {:.8}%".format(hack_round, cost * 100))
   hack_round+=1

#%%
img = hack_img[0]
img += 1.
img /= 2.
plt.imshow(img)
plt.imsave('hack_img', img)

#%%
print('Generating hacked test set...')
X_test_adver = np.zeros((X_test.shape[0], 299, 299, 3))

model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

for i in range(448, 600):
    print('Now hack test image index %s.' % i)
    start = datetime.datetime.now()
    ori_img = X_test[i]
    ori_img = np.expand_dims(ori_img, axis=0)
    hack_img = np.copy(ori_img)

    max_change_above = ori_img + 0.01
    max_change_below = ori_img - 0.01

    fake_cata = int(Y_test[i][0])

    cost_function = model_output_layer[0, fake_cata]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], 
                                                    [cost_function, gradient_function])
    
    e = 0.008
    cost = 0.0
    hack_round = 0
    while cost < 0.90:
       cost, gradients = grab_cost_and_gradients_from_model([hack_img, 0])
       g = np.sign(gradients)
       hack_img += g * e
       hack_img = np.clip(hack_img, max_change_below, max_change_above)
       hack_img = np.clip(hack_img, -1.0, 1.0)
       hack_round += 1
       if(hack_round > 50):
           print('Now index is %s, cost is %s' % (i, cost))
           break
       
    X_test_adver[i] = hack_img[0]
    end = datetime.datetime.now()
    print('Hack test image %s took %s seconds.' % (i, (end - start).seconds))

np.savez(DATA_DIR + 'dataset_car-airplane_adversarial_testset_448-600.npz', X_test_adver = X_test_adver, Y_test = clean_trainset['Y_test'])

#%%
f1 = np.load(DATA_DIR + 'dataset_car-airplane_adversarial_testset_0-143.npz')['X_test_adver']
f2 = np.load(DATA_DIR + 'dataset_car-airplane_adversarial_testset_144-300.npz')['X_test_adver']
f3 = np.load(DATA_DIR + 'dataset_car-airplane_adversarial_testset_301-447.npz')['X_test_adver']
f4 = np.load(DATA_DIR + 'dataset_car-airplane_adversarial_testset_448-600.npz')['X_test_adver']

#%%
X_test_adver = np.zeros((600, 299, 299, 3))
for i in range(0, 144):
    X_test_adver[i] = f1[i]
for i in range(144, 301):
    X_test_adver[i] = f2[i]
for i in range(301, 448):
    X_test_adver[i] = f3[i]
for i in range(448, 600):
    X_test_adver[i] = f4[i]

#%%
np.savez(DATA_DIR + 'dataset_car-airplane_adversarial_testset_90per.npz', X_test_adver = X_test_adver, Y_test = clean_trainset['Y_test'])






