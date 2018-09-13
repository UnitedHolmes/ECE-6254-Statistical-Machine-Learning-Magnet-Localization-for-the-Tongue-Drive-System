# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:20:57 2017

@author: Ghaith
"""

import numpy as np
from sepStatData import sepStatData as sep
from saveModel import saveModel
from getDataName import getDataName
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#from sklearn.linear_model import LassoCV as Lasso
from calcStats import calcStats
#from calcStatsOneAxis import calcStatsOneAxis
import os
"""
In this code, we eliminate features that have low coorelation with magnet
location

"""

##Select training file name and load it
fname = 'tcv_7_80.npy'    
f = 'Simulation data/tcv' + os.path.altsep + fname
data = np.load(f)
fname = fname.split('.')[0]
## Separate data
X,coord,count = sep(data) # X is nx12, coord is nx2

"""
Preprocessing
- First ocmpute the cross-correlation matrix between X (sensor measurements
 and coord (X, Y), this is a 14 by 14 matrixm but we are only concerned with
the last two rows as they show coorelation between measurements and magnet
location. 
-Then fit a regressor for X using those measurement components that have 
correlate with X above a certain threshold, i.e., abs(correlation coeff) is 
greater than a thershold xthershold
same thing with Y
then predict X and y separately 
""""" 
C=np.append(X,coord,1)
 
R =np.corrcoef(np.transpose(C))
""" R conains correlation coefficients for all input vectors but we all need
the last two rows corresponding to input with labels
"""
xthreshold=0.5
ythreshold=0.1
Xi=[]
Yi=[]
for i in range(12):
    if np.absolute(R[12,i])>xthreshold:
        Xi.append(i)
    if np.absolute(R[13,i])>ythreshold:
        Yi.append(i)
        
Xr=X[:,Xi]
Yr=X[:,Yi]
        

## Get cubic fit
plyX = PolynomialFeatures(3)
Xr = plyX.fit_transform(Xr) # nxd

plyY = PolynomialFeatures(3)
Yr = plyY.fit_transform(Yr) # nxd   
    
## Preprocess
scalerX = StandardScaler()
Xr = scalerX.fit_transform(Xr)

scalerY = StandardScaler()
Yr = scalerY.fit_transform(Yr)
    
## Train
lassoX = Lasso(cv=4,verbose =1, n_alphas=1000, max_iter=1500)
print('got here')
Xs=coord[:,0]
lassoX.fit(Xr,Xs)

lassoY = Lasso(cv=4,verbose =1, n_alphas=1000, max_iter=1500)
Ys=coord[:,1]
lassoY.fit(Yr,Ys)


"""
#save model
fModel, fScaler = getDataName(name_str = fname, dtype = 'static', tcvt = 'tcv', isModel = True)
fModelx=fModel+'X';
saveModel(fModelx, lassoX)
fModely=fModel+'Y';
saveModel(fModely, lassoY)
saveModel(fScaler, scaler)
"""

#testing
ftest = "test1.npy"
f = 'data/robot/test/' + os.path.altsep + ftest
Tdata = np.load(f) #nxd
XT,coordT,count = sep(Tdata)

XTr=XT[:,Xi]
YTr=XT[:,Yi]
## Get cubic fit
#ply = PolynomialFeatures(3)
XTr = plyX.transform(XTr) # nxd
YTr = plyY.transform(YTr) # nxd


## Scale testing data
XTr = scalerX.transform(XTr)
YTr = scalerY.transform(YTr)

## Predict
CoordX_pred = lassoX.predict(XTr)
CoordY_pred = lassoY.predict(YTr)

print("Test1: ")    
## Calculate the statistics
e,m,s,r = calcStatsOneAxis(coordT[:,0], CoordX_pred,isPrint=True)
e,m,s,r = calcStatsOneAxis(coordT[:,1], CoordY_pred,isPrint=True)

CoordXY_pred = np.vstack((CoordX_pred,CoordY_pred)).T
e,m,s,r = calcStats(coordT, CoordXY_pred,isPrint=True)
    
ftest = "test2.npy"
f = 'data/robot/test/' + os.path.altsep + ftest
Tdata = np.load(f) #nxd
XT,coordT,count = sep(Tdata)

XTr=XT[:,Xi]
YTr=XT[:,Yi]
## Get cubic fit
#ply = PolynomialFeatures(3)
XTr = plyX.transform(XTr) # nxd
YTr = plyY.transform(YTr) # nxd


## Scale testing data
XTr = scalerX.transform(XTr)
YTr = scalerY.transform(YTr)

## Predict
CoordX_pred = lassoX.predict(XTr)
CoordY_pred = lassoY.predict(YTr)
    
print("Test2: ")    

## Calculate the statistics
e,m,s,r = calcStatsOneAxis(coordT[:,0], CoordX_pred,isPrint=True)
e,m,s,r = calcStatsOneAxis(coordT[:,1], CoordY_pred,isPrint=True)

CoordXY_pred = np.vstack((CoordX_pred,CoordY_pred)).T
e,m,s,r = calcStats(coordT, CoordXY_pred,isPrint=True)

ftest = "test3.npy"
f = 'data/robot/test/' + os.path.altsep + ftest
Tdata = np.load(f) #nxd
XT,coordT,count = sep(Tdata)

XTr=XT[:,Xi]
YTr=XT[:,Yi]
## Get cubic fit
#ply = PolynomialFeatures(3)
XTr = plyX.transform(XTr) # nxd
YTr = plyY.transform(YTr) # nxd


## Scale testing data
XTr = scalerX.transform(XTr)
YTr = scalerY.transform(YTr)

## Predict
CoordX_pred = lassoX.predict(XTr)
CoordY_pred = lassoY.predict(YTr)

print("Test3: ")    

## Calculate the statistics
e,m,s,r = calcStatsOneAxis(coordT[:,0], CoordX_pred,isPrint=True)
e,m,s,r = calcStatsOneAxis(coordT[:,1], CoordY_pred,isPrint=True)

CoordXY_pred = np.vstack((CoordX_pred,CoordY_pred)).T
e,m,s,r = calcStats(coordT, CoordXY_pred,isPrint=True)
    
ftest = "test4.npy"
f = 'data/robot/test/' + os.path.altsep + ftest
Tdata = np.load(f) #nxd
XT,coordT,count = sep(Tdata)

XTr=XT[:,Xi]
YTr=XT[:,Yi]
## Get cubic fit
#ply = PolynomialFeatures(3)
XTr = plyX.transform(XTr) # nxd
YTr = plyY.transform(YTr) # nxd


## Scale testing data
XTr = scalerX.transform(XTr)
YTr = scalerY.transform(YTr)

## Predict
CoordX_pred = lassoX.predict(XTr)
CoordY_pred = lassoY.predict(YTr)
print("Test4: ")    

## Calculate the statistics
e,m,s,r = calcStatsOneAxis(coordT[:,0], CoordX_pred,isPrint=True)
e,m,s,r = calcStatsOneAxis(coordT[:,1], CoordY_pred,isPrint=True)

CoordXY_pred = np.vstack((CoordX_pred,CoordY_pred)).T
e,m,s,r = calcStats(coordT, CoordXY_pred,isPrint=True)
    
ftest = "test5.npy"
f = 'data/robot/test/' + os.path.altsep + ftest
Tdata = np.load(f) #nxd
XT,coordT,count = sep(Tdata)

XTr=XT[:,Xi]
YTr=XT[:,Yi]
## Get cubic fit
#ply = PolynomialFeatures(3)
XTr = plyX.transform(XTr) # nxd
YTr = plyY.transform(YTr) # nxd


## Scale testing data
XTr = scalerX.transform(XTr)
YTr = scalerY.transform(YTr)

## Predict
CoordX_pred = lassoX.predict(XTr)
CoordY_pred = lassoY.predict(YTr)
print("Test5: ")    

## Calculate the statistics
e,m,s,r = calcStatsOneAxis(coordT[:,0], CoordX_pred,isPrint=True)
e,m,s,r = calcStatsOneAxis(coordT[:,1], CoordY_pred,isPrint=True)

CoordXY_pred = np.vstack((CoordX_pred,CoordY_pred)).T
e,m,s,r = calcStats(coordT, CoordXY_pred,isPrint=True)
    

#    labels = lasso.predict(X)
    
#    RMSE,max_err,min_err,std = calcStats(coord, labels)
    

    