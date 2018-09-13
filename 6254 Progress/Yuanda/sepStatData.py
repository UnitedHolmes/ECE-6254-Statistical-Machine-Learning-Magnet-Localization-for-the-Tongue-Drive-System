# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:45:31 2016

@author: mjkagie
"""

def sepStatData(data):
    """
    Use to separate the dynamic data download
    
    Given:
        -data: nxd array
        
    Produce:
        -X: nx12 array of magnetic readings
        -coord: nx2 array of X,Y coords (labels)
        -count: nx2 array of trial counts, sample counts
    """
    X = data[:,:12]
    coord = data[:,12:14]
    count = data[:,14:].astype(int)
    return X,coord,count