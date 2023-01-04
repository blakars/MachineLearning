import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

forestdata  = pd.read_csv('ForestTypesData.csv'); # load data as pandas data frame 
classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
classidx    = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
K           = len(classlabels)               # number of classes 
T_txt = forestdata.values[:,0]               # array of class labels of data vectors (class label is first data attribute)
X = forestdata.values[:,1:]                  # array of feature vectors (features are remaining attributes)
T = [classidx[t.strip()] for t in T_txt]     # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
X,T = np.array(X,'float'),np.array(T,'int')  # explicitely convert to numpy array (otherwise there may occur type confusions/AttributeErrors...)

X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

layer = tf.keras.layers.Normalization()
layer.adapt(X_train)
X_train = layer(X_train)
X_test = layer(X_test)

mlp_model = Sequential()
mlp_model.add(Dense(units=5, input_dim=27, activation='tanh'))
lr_schedule= tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=1, decay_rate=0.01)
optimizer= SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False, name='SGD')
mlp_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
mlp_model.fit(X_train, T_train, batch_size=20, epochs=10, validation_split=0.2)
score = mlp_model.evaluate(X_test, T_test, verbose=0)
print(mlp_model.metrics_names)
print(score)