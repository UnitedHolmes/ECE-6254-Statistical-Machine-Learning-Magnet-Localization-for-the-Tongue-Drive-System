# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:33:36 2017

@author: NFLS_UnitedHolmes
"""

import numpy as np
from sepStatData import sepStatData as sep
from saveModel import saveModel
from getDataName import getDataName
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#from sklearn.linear_model import MultiTaskLassoCV as Lasso
from sklearn.neural_network import MLPClassifier
from calcStats import calcStats
from turn_y_into_binary import turn_y_into_binary
from turn_binary_into_y import turn_binary_into_y
#from getDataName import getDataName
#from calcStats import calcStats
import os
import time

t = time.time()

##Select training file name and load it
fname = 'tcv_7_80.npy'    
f = 'manual/tcv' + os.path.altsep + fname
data = np.load(f)
fname = fname.split('.')[0]
## Separate data
X,coord,count = sep(data) 
# X is column 1-12, coord is column 13-14, count is column 15-17

## Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)
coord = scaler.fit_transform(coord)
new_coord, train_coord_values, digits = turn_y_into_binary(coord)

## Neural Network Training
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15, ), random_state=1)
#y = np.zeros()
#y = complex(coord[:,1],coord[:,2])
clf.fit(X, new_coord)

elapsed_training = time.time() - t
print('Training Time: ', elapsed_training, ' sec')

##save model
#fModel, fScaler = getDataName(name_str = fname, dtype = 'static', tcvt = 'tcv', isModel = True)
#saveModel(fModel, clf)
#saveModel(fScaler, scaler)

##testing
ftest = "test1.npy"
f = 'manual/test/' + os.path.altsep + ftest
Tdata = np.load(f) #nxd
XT,coord2,count = sep(Tdata)
coord2 = scaler.fit_transform(coord2)
coord2 = np.round(coord2.dot(2))/2
new_coord2, test_coord_values, digits = turn_y_into_binary(coord2)
#
### Scale testing data
XT = scaler.fit_transform(XT)
#
### Predict
Y_pred = clf.predict(XT)
Y_pred_normalized = turn_binary_into_y(Y_pred, test_coord_values, digits)
#Y_pred = Y_pred.reshape(-1,1)
    
## Calculate the statistics
#e,m,s,r = calcStats(new_coord2, Y_pred,isPrint=True)
e,m,s,r = calcStats(coord2, Y_pred_normalized,isPrint=True)
elapsed = time.time() - t
print('Totoal Time: ', elapsed,' sec')