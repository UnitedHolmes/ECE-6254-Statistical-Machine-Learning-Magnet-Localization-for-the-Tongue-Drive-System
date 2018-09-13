# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:45:54 2017

@author: NFLS_UnitedHolmes
"""
## Single output method
## With 

import numpy as np
from sepStatData import sepStatData as sep
#from saveModel import saveModel
#from getDataName import getDataName
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import MultiTaskLassoCV as Lasso
from sklearn.neural_network import MLPClassifier
from calcStats import calcStats
from calcStatsOneAxis import calcStatsOneAxis
from turn_y_into_binary_new import turn_y_into_binary_new
from turn_binary_into_y_new import turn_binary_into_y_new
#from extend_y_pred import extend_y_pred
#from getDataName import getDataName
#from calcStats import calcStats
import os
import time
import xlsxwriter
#from openpyxl import Workbook

xthreshold=0.5
ythreshold=0.1

t = time.time()
for NN_layer in [5, 15, 25, 50, 100, 150, 200, 250, 300, 500, 750, 1000]:
    
    for data_type_ind in [1, 2]:
        if data_type_ind == 1:
            data_type = 'Experimental'
            print(' ')
            print(data_type)
            print(' ')
        elif data_type_ind == 2:
            data_type = 'Simulation'
            print(' ')
            print(data_type)
            print(' ')
        
        fexcel = data_type + ' Result with ' + str(NN_layer) + ' layers ' + str(xthreshold) + 'X' + str(ythreshold) + 'Y.xlsx'
        workbook = xlsxwriter.Workbook(fexcel)
        worksheet = workbook.add_worksheet('ANN')
        worksheet.write('A1', 'NN Layers = ' + str(NN_layer))
        worksheet.write('A2', 'Evaluation Data')
        worksheet.write('B2', 'Training Trial No')
        worksheet.write('C2', 'No points for each trial')
        worksheet.write('D2', 'RMSE')
        worksheet.write('E2', 'RMSE-X')
        worksheet.write('F2', 'RMSE-Y')
        worksheet.write('G2', 'Max Error')
        worksheet.write('H2', 'std')

        ##Select training file name and load it
        training_trials = range(1, 16)
        for train_ind in training_trials:
            if train_ind <= 8:
                ftrain = 'tcv_1_' + str(train_ind) + '0.npy'
                B_entry = '1'
                C_entry = str(train_ind) + '0'
            elif train_ind < 15:
                ftrain = 'tcv_' + str(train_ind-7) + '_10.npy'
                B_entry = str(train_ind-7)
                C_entry = '10'
            elif train_ind == 15:
                ftrain = 'tcv_7_80.npy'
                B_entry = '7'
                C_entry = '80'
    
            time_training_start = time.time()
            f = data_type + ' data/tcv' + os.path.altsep + ftrain
            data = np.load(f)
#            ftrain = ftrain.split('.')[0]
            ## Separate data
            X,coord,count = sep(data) 
            if data_type_ind == 1:
                    coord = coord.dot(10)
            # X is column 1-12, coord is column 13-14, count is column 15-17
            
            ## Preprocess
            
            C=np.append(X,coord,1)
 
            R =np.corrcoef(np.transpose(C))
            """ R conains correlation coefficients for all input vectors but we all need
            the last two rows corresponding to input with labels
            """
            
            Xi=[]
            Yi=[]
            for i in range(12):
                if np.absolute(R[12,i])>xthreshold:
                    Xi.append(i)
                if np.absolute(R[13,i])>ythreshold:
                    Yi.append(i)
        
            Xr=X[:,Xi]
            Yr=X[:,Yi]

            scalerX = StandardScaler()
            scalerY = StandardScaler()            
            Xr = scalerX.fit_transform(Xr)
            Yr = scalerY.fit_transform(Yr)            
#            coord = scaler.fit_transform(coord)
            
#            new_coord, train_coord_values, digits_train = turn_y_into_binary_new(coord)
    
            ## Neural Network Training
            clfX = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(NN_layer, ), random_state=1)
            clfY = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(NN_layer, ), random_state=1)
            #y = np.zeros()
            #y = complex(coord[:,1],coord[:,2])
            clfX.fit(Xr, coord[:,0])
            clfY.fit(Yr, coord[:,1])
    
            time_training_end = time.time() - time_training_start
            print('Training for ' + ftrain + ' has finished!')
            print('Training Time: ', time_training_end, ' sec')
    
            ##save model
            #fModel, fScaler = getDataName(name_str = ftrain, dtype = 'static', tcvt = 'tcv', isModel = True)
            #saveModel(fModel, clf)
            #saveModel(fScaler, scaler)
    
            ##testing
            for test_ind in range(1,6):
                ftest = 'test' + str(test_ind) + '.npy'
                print(ftest)
                f = data_type + ' data/test/' + os.path.altsep + ftest
                Tdata = np.load(f) #nxd
                XT,coord2,count = sep(Tdata)
                if data_type_ind == 1:
                    coord2 = coord2.dot(10)
                
#                new_coord2, test_coord_values, digits_test = turn_y_into_binary_new(coord2)
    
                ### Scale testing data
                XTr = XT[:,Xi]
                YTr = XT[:,Yi]                
                XTr = scalerX.fit_transform(XTr)
                YTr = scalerY.fit_transform(YTr)
    
                ### Predict
                Y_pred1 = clfX.predict(XTr)
                Y_pred2 = clfY.predict(YTr)
#                coord2_pred = np.concatenate((Y_pred1, Y_pred2), axis=0)
                coord2_pred = np.vstack((Y_pred1, Y_pred2))
                coord2_pred = coord2_pred.T
    
                ## Calculate the statistics
                #e,m,s,r = calcStats(new_coord2, Y_pred,isPrint=True)                
                e,m,s,r = calcStats(coord2, coord2_pred,isPrint=False)
                line_to_write = (test_ind - 1) * len(training_trials) + train_ind + 2
                print('line to write: ', line_to_write)
                print(' ')
                if line_to_write == (test_ind - 1) * len(training_trials) + 3:
                    worksheet.write('A' + str(line_to_write), 'test' + str(test_ind))
                worksheet.write('B' + str(line_to_write), B_entry)
                worksheet.write('C' + str(line_to_write), C_entry)
                worksheet.write('D' + str(line_to_write), str(r))               
                worksheet.write('G' + str(line_to_write), str(m))
                worksheet.write('H' + str(line_to_write), str(s))
                
                e,m,s,r = calcStatsOneAxis(coord2[:,0], Y_pred1,isPrint=False)
                worksheet.write('E' + str(line_to_write), str(r))
                e,m,s,r = calcStatsOneAxis(coord2[:,1], Y_pred2,isPrint=False)
                worksheet.write('F' + str(line_to_write), str(r))
        
                elapsed = time.time() - time_training_start

            print('Total Time for ' + ftrain + ' : ', elapsed,' sec')

        workbook.close()
t_end = time.time() - t
print('Final Time: ', t_end, ' sec')