# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:20:16 2017

@author: NFLS_UnitedHolmes
"""

import numpy as np
def calcStatsOneAxis(y_true, y_pred,isPrint=False):
    """
    Calculate the statistics of the classifier
    
    Given:
        -y_true: nx2 vector of true y values
        -y_pred: nx2 vector of predicted y values
        
    Return:
        -mean error, max error, standard deviation, rmse
    """
        
    
    ## Calculate errors
    dist = np.sqrt(np.sum((y_true - y_pred)**2,axis=0))
    
    ## Mean error
    e = np.mean(dist)
    
    ## Max error
    m = np.max(dist)
    
    ## Standard Deviation
    s = np.std(dist)
    
    ## RMSE
    r = np.sqrt(np.mean(((y_true - y_pred)**2)))
    
    
    if isPrint:
        print( 'Mean Error: %f' % e)
        print( 'Max Error: %f' % m)
        print( 'Standard Deviation: %f' % s)
        print( 'RMSE: %f' % r)
    
    return e,m,s,r
