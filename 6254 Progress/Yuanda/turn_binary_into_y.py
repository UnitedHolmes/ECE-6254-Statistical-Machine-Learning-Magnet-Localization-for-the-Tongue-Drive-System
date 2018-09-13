# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:17:27 2017

@author: NFLS_UnitedHolmes
"""

import numpy as np
import math
def turn_binary_into_y(y, coord_values, digits):
    rows, columns = np.shape(y)
    output_columns2 = np.round(columns/digits)
    output_columns = 0
    while output_columns < output_columns2:
        output_columns = output_columns + 1
    output_y = np.zeros((rows,output_columns))
    
    ## entend coord_value array
    coord_len = len(coord_values)
    while coord_len < math.pow(2,digits):
        coord_values.extend([coord_values[coord_len-1]*2-coord_values[coord_len-2]])
        coord_len = len(coord_values)
    
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

    return output_y