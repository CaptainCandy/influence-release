# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:36:00 2019

@author: Tang
"""

import numpy as np
from matplotlib import pyplot as plt

def change_brightness(img, n):
    '''
    img: original image (x, x, 3)
    n: positive or negative int, brightness to add
    '''
    
    img = img.astype(np.int32)
    for i in range(3):
        img[:,:,i] += n
    
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    
    return img

#%%
ori_set = np.load('C:\\Tang\\data\\car_airplane\\dataset_car-airplane_train-1000_test-300.npz')
ori_Xtest = ori_set['X_test']
ori_Ytest = ori_set['Y_test']
bright_Xtest = np.zeros((600, 299, 299, 3))
dark_Xtest = np.zeros((600, 299, 299, 3))

#%%
for i, img in enumerate(ori_Xtest):
    print('brit/dark image %s' % i)
    img = (ori_Xtest[i]+1)/2*255
    img.flags.writeable = True
    bright = change_brightness(img, +100)
    dark = change_brightness(img, -100)
    
    bright_Xtest[i] = bright/255.0*2.0-1.0
    dark_Xtest[i] = dark/255.0*2.0-1.0

#%%
np.savez('C:\\Tang\\data\\car_airplane\\dataset_car-airplane_bright_testset.npz', X_test = bright_Xtest, Y_test = ori_Ytest)
np.savez('C:\\Tang\\data\\car_airplane\\dataset_car-airplane_dark_testset.npz', X_test = dark_Xtest, Y_test = ori_Ytest)

#%%
img = (ori_Xtest[523]+1.)/2.
#img = (bright_Xtest[523]+1.)/2.
#img = (dark_Xtest[523]+1.)/2.
plt.imshow(img)