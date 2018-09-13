# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:35:04 2016

@author: mjkagie
"""
import os
def getDataName(name_str, dtype,tcvt,isModel=False):
    """
    Function to keep naming conventions the same throughout
    Given:
        -name_str: name of the file
        -dtype: manual,dynamic,robot,model,rcirc
        -tcvt: tcv or test
        -isModel: whether we are saving a model or an array
    Return:
        -fname
    """
    folder = {'manual': 'data/manual', 'dynamic': 'data/dynamic','model': 'model','robot':'data/robot','rcirc':'data/robotic_circle'}
    ftcvt = {'tcv':'tcv','test':'test'}
    endings = {False:'.npy',True:'.pkl'}
    s = os.path.sep
    
    
    if isModel:
        fname = folder['model'] + s + dtype + '_' + name_str
        fModel = fname + '_model' + endings[isModel]
        fScaler = fname + '_scaler' + endings[isModel]
        return fModel, fScaler
    else:
        fname = folder[dtype] + s + ftcvt[tcvt] + s + name_str + endings[isModel]
        return fname