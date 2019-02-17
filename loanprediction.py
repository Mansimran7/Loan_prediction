# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 01:29:03 2019

@author: Mansimran Anand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Train.csv')


#taking are of missing data 

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

x_train = dataset.iloc[:,7:9]
y_train = dataset.iloc[:,12:13]

imputer = imputer.fit(x_train)
x_train= imputer.transform(x_train)

dataset.describe()
c=0
for i in len(x_train):
    if x_train[8][i]=='nan':
        c+=1
x_train


