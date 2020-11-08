# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 08:51:18 2020

@author: Abdul Qayyum
"""

#%% ############################# Dataset prepartion from depression dataset ########################
from keras.models import Sequential
from keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense

# Control Dataset
import os
import scipy
from scipy import io, signal
import matplotlib.pyplot as plt
#import dtcwt
import numpy as np
import itertools
import pywt
#from __future__ import print_function
from matplotlib import pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

 ############################################# MDD class dataset #########################################  
# load dataset for control class
dataDir = "D:\\Drwajidwork\\Data for Qayyum\\Data for Qayyum\\controls\\"
#dir_seg = dataDir + "/Ahmad/"
#dir_segEO=os.listdir(dir_seg)[0]
#dir_segEC=os.listdir(dir_seg)[1]
# function for normalization feature matrix
#def numfun(f):
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    scaledtrain = scaler.fit_transform(f)
#    return scaledtrain
#Predicted = dir_data + "/preds_8300128res/"
matsEOControl = []
matsECControl = []
dataec=[]
for file in os.listdir( dataDir ) :
    EO=os.listdir(dataDir+file)[0]
    EC=os.listdir(dataDir+file)[1]
    matsEOControl.append( scipy.io.loadmat(os.path.join(dataDir+file,EO )))
    matsECControl.append(scipy.io.loadmat(os.path.join(dataDir+file,EC ) ))
    
    
    
# ###################################### make time windows of  EC dataset ####################################
from numpy import array
n=15000 # total number of samples
samples = list()
length = 500 # segmnet length
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecControlEC=[]
y=[]
for dd in matsECControl:
    vv=dd['data']
    vv1=vv.transpose()
    samples1=conactfun(vv1)
    dataControlEC = array(samples)
    dataecControlEC.append(dataControlEC) 
    y.append(1)
 
# ################################################## make time windows of  EO dataset ##########################
from numpy import array
n=15000
samples = list()
#length = 500
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecControlEO=[]
y=[]
for dd in matsEOControl :
    vv=dd['data']
    vv1=vv.transpose()
    samples1=conactfun(vv1)
    dataControlEO = array(samples)
    dataecControlEO.append(dataControlEO) 
    y.append(1)    
    
    
############################################# MDD class dataset #########################################    

#% make class2 MDD dataset class
import os
import scipy
from scipy import io, signal
import matplotlib.pyplot as plt
#import dtcwt
import numpy as np
import itertools
import pywt
# load dataset for MDD class
dataDir = "D:\\Drwajidwork\\Data for Qayyum\\Data for Qayyum\\MDD\\"
#dir_segEO=os.listdir(dir_seg)[0]
#dir_segEC=os.listdir(dir_seg)[1]

#Predicted = dir_data + "/preds_8300128res/"
matsEOMDD = []
matsECMDD = []
dataec=[]

for file in os.listdir( dataDir ) :
    EO=os.listdir(dataDir+file)[0]
    EC=os.listdir(dataDir+file)[1]
    matsEOMDD.append( scipy.io.loadmat(os.path.join(dataDir+file,EO )))
    matsECMDD.append(scipy.io.loadmat(os.path.join(dataDir+file,EC ) ))
    
    
    
# ################################### make window for MMD class for EC dataset ##########################
from numpy import array
n=15000 # total number of time samples in the dataset
samples = list()
#length = 500
y=[]
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecMDDEC=[]
for dd in matsECMDD:
    vv=dd['data']
    vv1=vv.transpose()
    samples1=conactfun(vv1)
    dataMDDEC = array(samples)
    dataecMDDEC.append(dataMDDEC)
    y.append(0)

# Data for EO for MDD class
############################################ make window for MMD class for EO dataset ################## 
from numpy import array
n=15000
samples = list()
#length = 500
y=[]
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecMDDEO=[]
for dd in matsEOMDD:
    vv=dd['data']
    vv1=vv.transpose()
    samples1=conactfun(vv1)
    dataMDDEO = array(samples)
    dataecMDDEO.append(dataMDDEO)
    y.append(0)   

dataControlEC.shape[0]
############################################## define labels and feature matrix for EC and EO  ############################
# For control dataset create labels
yoneECC=np.array(np.ones(dataControlEC.shape[0]))
yoneEOC=np.array(np.ones(dataControlEO.shape[0]))
yzeroECC=np.array(np.zeros(dataControlEC.shape[0]))
yzeroEOC=np.array(np.zeros(dataControlEO.shape[0]))
# for MDD dataset create labels
# For control dataset create labels
yoneECM=np.array(np.ones(dataMDDEC.shape[0]))
yoneEOM=np.array(np.ones(dataMDDEO.shape[0]))
yzeroECM=np.array(np.zeros(dataMDDEC.shape[0]))
yzeroEOM=np.array(np.zeros(dataMDDEO.shape[0]))

# concatenate Labels for both classes label for control class, y22 for MMD class
labelsEC=np.concatenate((yoneECC,yzeroECM),axis=0)
# concatenate dataset for EC for two classes(control and MMD).   
DatamatrixEC=np.concatenate((dataControlEO,dataMDDEO),axis=0)
X_shortEC=DatamatrixEC
y_shortEC=labelsEC

# concatenate for Eye close dataset for two classes
labelsEO=np.concatenate((yoneEOC,yzeroEOM),axis=0)
# concatenate dataset for EC for two classes(control and MMD).   
DatamatrixEO=np.concatenate((dataControlEC,dataMDDEC),axis=0)
X_shortEO=DatamatrixEO
y_shortEO=labelsEO

################################################### dataset spliting into training and testing #####################

# create for eye open(EO) ######################### split dataset into training and testing ####################
import random
seed=42
random.seed(seed)
# shuffled dataset for training and testing
from sklearn.model_selection import StratifiedShuffleSplit

# use strat. shuffle split to get indices for test and training data 
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
sss.get_n_splits(X_shortEO, y_shortEO)

# create train and test dataset for classification
# take the indices generated by stratified shuffle split and make the test and training datasets
for train_index, test_index in sss.split(X_shortEO, y_shortEO):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_trainEO, X_testEO = X_shortEO[train_index], X_shortEO[test_index]
    y_trainEO, y_testEO = y_shortEO[train_index], y_shortEO[test_index]
    
#  ##################################### convert into onehot encoding ############################
y_train_hotEO = np_utils.to_categorical(y_trainEO, 2)
y_test_hotEO=np_utils.to_categorical(y_testEO, 2)


# create for EC ################################### split dataset into training and testing ###############

import random
seed=42
random.seed(seed)
# shuffled dataset for training and testing
from sklearn.model_selection import StratifiedShuffleSplit

# use strat. shuffle split to get indices for test and training data 
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
sss.get_n_splits(X_shortEC, y_shortEC)

# create train and test dataset for classification
# take the indices generated by stratified shuffle split and make the test and training datasets
for train_index, test_index in sss.split(X_shortEC, y_shortEC):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_trainEC, X_testEC = X_shortEC[train_index], X_shortEC[test_index]
    y_trainEC, y_testEC = y_shortEC[train_index], y_shortEC[test_index]
    

##################################### convert labels into onehot encoding ############################
y_train_hotEC = np_utils.to_categorical(y_trainEC, 2)
y_test_hotEC=np_utils.to_categorical(y_testEC, 2)

############################################## saving training and testing set in npy array ################
import numpy as np

#################################################EC dataset for training and testing #################
np.save('TrainingEC',X_trainEC)
np.save('TestingEC',X_testEC)

np.save('TraininglabelsEC',y_train_hotEC)
np.save('TestinglabelsEC',y_test_hotEC)

#################################################EO dataset for training and testing #################

np.save('TrainingEO',X_trainEO)
np.save('TestingEO',X_testEO)

np.save('TraininglabelsEO',y_train_hotEO)
np.save('TestinglabelsEO',y_test_hotEO)












