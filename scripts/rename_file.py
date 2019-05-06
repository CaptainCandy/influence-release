# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:37:52 2019

@author: Administrator
"""

import os
path = 'C:/Tang/influence-release-master/scripts/output'
i = 50
for file in os.listdir(path):
    if file=='rbf_carair_results_%s.npz' % i:
        print(file)
        newname = 'rbf_dogfish_results_%s.npz' % i
        os.rename(os.path.join(path,file),os.path.join(path,newname))
        
#%%