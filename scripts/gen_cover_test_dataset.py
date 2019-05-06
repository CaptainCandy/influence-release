# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:19:59 2019

@author: Tang
"""

import numpy as np
import os
from PIL import Image

#%%
ori_set = np.load('C:\\Tang\\data\\car_airplane\\dataset_car-airplane_train-1000_test-300.npz')
ori_Xtest = ori_set['X_test']
ori_Ytest = ori_set['Y_test']
index = [15, 18, 38, 41, 59, 91, 99, 103, 111, 116, 123, 143, 147, 165, 167, 168, 
         176, 179, 205, 243, 268, 280, 324, 359, 365, 406, 522, 533, 549, 576, 583]

#%%
for i, idx in enumerate(index):
    ori = Image.fromarray(np.uint8((ori_Xtest[idx]+1)/2*255))
    ori.save('C:\\Tang\\data\\car_airplane\\cover\\%s.jpg' % i)

#%%
files = os.listdir('C:\\Tang\\data\\car_airplane\\cover\\done\\')
cover_Xtest = np.zeros((len(files), 299, 299, 3))
cover_Ytest = np.zeros((len(files)))
for i, img in enumerate(files):
    cover = Image.open('C:\\Tang\\data\\car_airplane\\cover\\done\\' + img)
    cover = np.array(cover)
    cover_Xtest[i] = cover/255.*2.-1.
    print(img, img.split('.')[0].split('_')[0])
    if img.split('.')[0].split('_')[0] in ['7', '18', '19']:
        cover_Ytest[i] = 1

#%%
np.savez('C:\\Tang\\data\\car_airplane\\dataset_car-airplane_cover_testset.npz', 
         X_test = cover_Xtest,
         Y_test = cover_Ytest)