# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:00:56 2020

@author: AVINASH SHUKLA
"""


#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#%% encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x1 = LabelEncoder()
X[:, 1] = labelencoder_x1.fit_transform(X[:, 1])

labelencoder_x2 = LabelEncoder()
X[:, 2] = labelencoder_x2.fit_transform(X[:, 2])

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",
         OneHotEncoder(),
         [1]
         )
        ],
    remainder='passthrough'
    )
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')
X = X[:, 1:]

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#%%making ANN
classifier = Sequential()

#%% making hidden layer and input layer
classifier.add(Dense(units = 6,kernel_initializer='uniform',activation='relu',input_dim = 11))
classifier.add(Dropout(rate = 0.1))

#%% making second hidden layer
classifier.add(Dense(units = 6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate = 0.1))

#%% output layer
classifier.add(Dense(units = 1,kernel_initializer='uniform',activation='sigmoid'))

#%% compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#%% fitting train set to ann
classifier.fit(X_train,y_train,batch_size=10,epochs=100) 

#%% predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#%% check test accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))
    











































