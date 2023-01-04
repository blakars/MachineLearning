#!/usr/bin/env python
# Programmgeruest zu IAS, Versuch 3, Aufgabe 4)
import numpy as np
import pandas as pd
import time
#from time import clock
from random import randint
from V3A3_MLP3Classifier import *
from V2A2_Regression import *

# (i) Define and construct MLP
print("\n(I) Define and construct MLP:")
M=4                    # number of hidden units
flagsBiasUnits=1       # bias units in input layer and hidden layer?
lmbda=1                # regularization coefficient
eta0=0.65               # initial learning rate
eta_fade=1./10          # fading factor for decreasing learning rate (e.g., 1/50 means after 50 epochs is learning rate half the initial value...)
maxEpochs=300           # max. number of learning epochs
nTrials = 1            # number of learning trials
eps = 1e-4             # stop learning if (normalized) error function becomes smaller than eps
debug = 1              # if >0 then debug mode: 1 = print Error, mean weight; 2=additionally check gradients; 3=additionally print weights
flagScaleData = 1      # if >0 then scale data vectors in X
mlp = MLP3Classifier(M,flagsBiasUnits,lmbda,eta0,eta_fade,maxEpochs,nTrials,eps,debug)

# (ii) Load data and preprocessing
print("\n(II) Load data and do preprocessing:")
forestdata  = pd.read_csv('../DATA/ForestTypes/ForestTypesData.csv'); # load data as pandas data frame
classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
classidx    = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
K           = len(classlabels)               # number of classes 
T_txt = forestdata.values[:,0]               # array of class labels of data vectors (class label is first data attribute)
X = forestdata.values[:,1:]                  # array of feature vectors (features are remaining attributes)
T = [classidx[t.strip()] for t in T_txt]     # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
X,T = np.array(X,'float'),np.array(T,'int')  # explicitely convert to numpy array (otherwise there may occur type confusions/AttributeErrors...)
if flagScaleData>0:
    scaler = DataScaler(X)
    X=scaler.scale(X)
N,D=X.shape                           # size and dimensionality of data set
print("Data set 'ForestData' has size N=", N, " and dimensionality D=",D, " and K=", K, " different classes")
print("X[0..9]=\n",X[0:10])
print("T_txt[0..9]=\n",T_txt[0:10])
print("T[0..9]=\n",T[0:10])

# (II) Test MLP with S-fold cross validation
S=3
t1=time.time()                            # start time
print('Time t1:',t1)
pE,pCE = mlp.crossvalidate(S,X,T)     # do S-fold cross validation for data X,T
t2=time.time()                            # end time
print('Time t2:',t2)
time_comp=t2-t1                       # computing time in seconds
print("\nS=",S," fold cross validation using the MLP yields the following results:")
print("Classification error probability = ", pE)
print("Accuracy = ", 1.0-pE)
print("Confusion Error Probabilities p(class i|class j) = \n", pCE)
print("Computing time = ", time_comp, " sec")

