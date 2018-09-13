# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:46:34 2017

@author: NFLS_UnitedHolmes
"""

import numpy as np
import math
def extend_y_pred(Y_pred, train_coord_values, digits_train, test_coord_values, digits_test):
    rows, columns = np.shape(Y_pred)
    output_y = np.zeros(rows, digits_test*2)
    
    for k in range(0, rows):
        for kk in range(0, output_columns):
            ind = 0
            for kkk in range(digits-1, -1, -1):
                if kk*digits+kkk >= columns:
                    print('Error!')
                else:
                    ind = ind + y[k, kk*digits+kkk]*math.pow(2,digits-kkk-1)
            
            ind2 = 0
            while ind2 < ind:
                ind2 = ind2 + 1
#            if ind2 > 6:
#                print(k, kk, kkk)
                
            output_y[k,kk] = coord_values[ind2]