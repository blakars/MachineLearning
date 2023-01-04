import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

forestdata  = pd.read_csv('ForestTypesData.csv'); # load data as pandas data frame 
classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
classidx    = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
K           = len(classlabels)               # number of classes 
T_txt = forestdata.values[:,0]               # array of class labels of data vectors (class label is first data attribute)
X = forestdata.values[:,1:]                  # array of feature vectors (features are remaining attributes)
T = [classidx[t.strip()] for t in T_txt]     # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
X,T = np.array(X,'float'),np.array(T,'int')  # explicitely convert to numpy array (otherwise there may occur type confusions/AttributeErrors...)


pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(5), solver='sgd', activation='tanh', alpha=5, learning_rate='adaptive', learning_rate_init=0.3, max_iter=250))])
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, T, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print(n_scores)