# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:42:45 2017

@author: NFLS_UnitedHolmes
"""


import numpy as np
import math
def turn_y_into_binary_new(y):
    # input must be normalized
    # output is a matrix of binary entries
    rows, columns = np.shape(y)
    coord_values = [];
    for k in range(-40, 50, 5):
        coord_values.extend([k])
        
    digits = math.ceil(math.log(len(coord_values), 2))
    
    ## make a transform table
#    for k in range(0, digits):
#        binary_array
#            
    output_y = np.zeros((rows,columns*digits))
    ## start transform
    for k in range(0, rows):
        for kk in range(0, columns):
            ind = coord_values.index(y[k,kk])
            #current_digit_ind = digits
            for kkk in range(digits-1, -1, -1):
                output_y[k,kk*digits+kkk] = ind % 2
                ind = math.floor(ind / 2)
                
    return output_y, coord_values, digits