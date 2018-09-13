# -*- coding: utf-8 -*-
"""
Convert data from .mat to .npy
Created on Wed Jun  1 17:11:47 2016

@author: mjkagie
"""
## Useful
ex_test = {'mat':'Simulation data/test', 'npy':'Simulation data/test'}
ex_tcv = {'mat':'Simulation data/tcv', 'npy':'Simulation data/tcv'}
#dynamicFolder = {'mat':'../data/processed_data/dynamic_circle/test','npy':'data/dynamic'}
#rcirc = {'mat':'../data/processed_data/robotic_circle','npy':'data/robotic_circle/test'}
folder = {'ex_test': ex_test, 'ex_tcv':ex_tcv}

## settings
ftype = 'ex_test'



import os
from scipy.io import loadmat,whosmat
import numpy as np
for filename in os.listdir(folder[ftype]['mat']):
    f = folder[ftype]['mat'] + os.path.sep + filename
    X = loadmat(f)
    if len(whosmat(f)) > 1:
        print(filename)
    arrName = whosmat(f)[0][0]
    data = X[arrName]
    
    fname = filename.split('.')[0]
    f = folder[ftype]['npy'] + os.path.sep + fname
    np.save(f, data)
