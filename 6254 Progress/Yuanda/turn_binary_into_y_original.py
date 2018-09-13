# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:40:30 2017

@author: NFLS_UnitedHolmes
"""

import numpy as np
import math
def turn_binary_into_y_original(y, original_coord2, digits):
    rows, columns = np.shape(y)
    output_columns2 = np.round(columns/digits)
    output_columns = 0
    while output_columns < output_columns2:
        output_columns = output_columns + 1
    output_y = np.zeros((rows,output_columns))
    
    copy_coord2 = np.copy(original_coord2)
    coord_values = []
    
    while ((copy_coord2 != 100).any()):   #sum(sum(y)) < 100*rows*columns:
        min_coord = copy_coord2.min()
        copy_coord2[copy_coord2 == min_coord] = 100
        coord_values.extend([min_coord])
#        count_values = count_values + 1
    
    ## entend coord_value array
    coord_len = len(coord_values)
    while coord_len < math.pow(2,digits):
        coord_values.extend([coord_values[coord_len-1]*2-coord_values[coord_len-2]])
        coord_len = len(coord_values)
    
    for k in range(0, rows):
        for kk in range(0, output_columns):
            ind = 0
            for kkk in range(digits-1, -1, -1):
                ind = ind + y[k, kk*digits+kkk]*math.pow(2,digits-kkk-1)
            
            ind2 = 0
            while ind2 < ind:
                ind2 = ind2 + 1
#            if ind2 > 6:
#                print(k, kk, kkk)
                
            output_y[k,kk] = coord_values[ind2]

    return output_y