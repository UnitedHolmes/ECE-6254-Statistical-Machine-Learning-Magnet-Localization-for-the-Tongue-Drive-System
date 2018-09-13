# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:57:21 2016

@author: mjkagie
"""
from sklearn.externals import joblib
def saveModel(fname, model):
    """
    Save a model
    
    Given:
        -fname: filename
        -model: model to save
    """
    joblib.dump(model, fname)