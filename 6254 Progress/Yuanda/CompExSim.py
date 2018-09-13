# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 20:24:54 2017

@author: NFLS_UnitedHolmes
"""

import os
import numpy as np
from sepStatData import sepStatData as sep

training_trials = range(1, 16)
for train_ind in training_trials:
    if train_ind <= 8:
        ftrain = 'tcv_1_' + str(train_ind) + '0.npy'        
    elif train_ind < 15:
        ftrain = 'tcv_' + str(train_ind-7) + '_10.npy'        
    elif train_ind == 15:
        ftrain = 'tcv_7_80.npy'
               
#    f_train_simulation = 'Simulation data/tcv' + os.path.altsep + ftrain
    f_train_simulation = 'manual/tcv' + os.path.altsep + ftrain
    data_train_simulation = np.load(f_train_simulation)
    X_train_simulation,coord_train_simulation,count = sep(data_train_simulation) 
    
    f_train_experimental = 'Experimental data/tcv' + os.path.altsep + ftrain
    data_train_experimental = np.load(f_train_experimental)
    X_train_experimental,coord_train_experimental,count = sep(data_train_experimental) 
    coord_train_experimental = coord_train_experimental.dot(10)
    
    if np.all(X_train_simulation==X_train_experimental):
        print('X are the same for ', str(train_ind), ' Trial')
        
    if np.all(coord_train_simulation==coord_train_experimental):
        print('Coord are the same for ', str(train_ind), ' Trial')
        
for test_ind in range(1,6):
    ftest = 'test' + str(test_ind) + '.npy'
    
    ftest_sim = 'Simulation data/test/' + os.path.altsep + ftest
    Ttest_sim = np.load(ftest_sim) #nxd
    XTtest_sim,coord2test_sim,count = sep(Ttest_sim)
    
    ftest_exp = 'Experimental data/test/' + os.path.altsep + ftest
    Ttest_exp = np.load(ftest_exp) #nxd
    XTtest_exp,coord2test_exp,count = sep(Ttest_exp)
    coord2test_exp = coord2test_exp.dot(10)
    
    if np.all(XTtest_sim == XTtest_exp):
        print('XT are the same for ', str(test_ind), ' test')
        
    if np.all(coord2test_sim == coord2test_exp):
        print('coord2 are the same for ', str(test_ind), ' test')