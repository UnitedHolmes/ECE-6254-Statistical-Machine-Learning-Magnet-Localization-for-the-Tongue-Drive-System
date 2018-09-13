# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:21:30 2017

@author: NFLS_UnitedHolmes
"""

import numpy as np
import math
def turn_y_into_binary(y):
    # input must be normalized
    # output is a matrix of binary entries
    rows, columns = np.shape(y)
    count_values = 0; # count of how many different normalized values are in y
#    y = np.round(y.dot(2))/2
    copy_y = np.copy(y)
    min_y_array = []
    
    ## To count for count_values
    while ((copy_y != 100).any()):   #sum(sum(y)) < 100*rows*columns:
        min_y = copy_y.min()
        copy_y[copy_y == min_y] = 100
        min_y_array.extend([min_y])
        count_values = count_values + 1
        
    digits = math.ceil(math.log(count_values, 2))
    
    ## make a transform table
#    for k in range(0, digits):
#        binary_array
#            
    output_y = np.zeros((rows,columns*digits))
    ## start transform
    for k in range(0, rows):
        for kk in range(0, columns):
            ind = min_y_array.index(y[k,kk])
            #current_digit_ind = digits
            for kkk in range(digits-1, -1, -1):
                output_y[k,kk*digits+kkk] = ind % 2
                ind = math.floor(ind / 2)
                
    return output_y, min_y_array, digits