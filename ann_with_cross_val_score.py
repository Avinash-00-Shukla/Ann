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


#%% EValuating the ann using k_folds
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def BuildClassifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,kernel_initializer='uniform',activation='relu',input_dim = 11))
    classifier.add(Dense(units = 6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units = 1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = BuildClassifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10,n_jobs=-1)
mean = accuracies.mean()
var = accuracies.std()
print('mean',mean)
print('variance',var)
    











































